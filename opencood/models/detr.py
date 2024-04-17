import torch.nn as nn
import torch
from opencood.models.detr_sub.models.backbone import build_backbone
from opencood.models.detr_sub.models.transformer import build_transformer
import torch.nn.functional as F
from opencood.models.detr_sub.util.misc import (nested_tensor_from_tensor_list,
                                       accuracy, get_world_size, interpolate,
                                       is_dist_avail_and_initialized)
class DETR(nn.Module):
    def __init__(self, args: dict):  # arg要包括backbone, transformer, num_classes, num_queries, aux_loss=False
        super(DETR, self).__init__()

        self.num_queries = args['num_queries']
        self.transformer = build_transformer(args)
        hidden_dim = self.transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, args['num_classes'] + 1)         # 类别预测
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)  # 边界框预测 输出维度4 (cx, cy, w, h) -> 类pixor 6维
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)  # object queries
        self.backbone = build_backbone(args)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)  # 一个卷积层，通常用于将从backbone（如ResNet）提取出的特征图投影（即转换）到Transformer所需的维度
        self.aux_loss = args["aux_loss"]

    def forward(self, data_dict):  # processed_data['ego']

        bev_input = data_dict['processed_lidar']['bev_input']

        # 根据原文，需要变成NestedTensor
        if isinstance(bev_input, (list, torch.Tensor)):
            bev_input = nested_tensor_from_tensor_list(bev_input)
        # Using the backbone to process the BEV input and extract features

        features, pos = self.backbone(bev_input)

        # Assuming the deepest feature map is relevant for the detection task
        src, mask = features[-1].decompose()  # Decomposing the deepest feature map
        assert mask is not None

        # Processing through the transformer with the src, mask, query embeddings, and positional embeddings
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # Generating classification and bounding box predictions from the transformer output
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        # Constructing the output dictionary with predictions
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            # Including auxiliary outputs if auxiliary losses are activated
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
