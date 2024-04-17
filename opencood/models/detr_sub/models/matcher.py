# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def obb_distance(self, obb1, obb2):
        # Extract components
        cos1, sin1, dx1, dy1, w1, l1 = obb1[:, 0], obb1[:, 1], obb1[:, 2], obb1[:, 3], obb1[:, 4], obb1[:, 5]
        cos2, sin2, dx2, dy2, w2, l2 = obb2[:, 0], obb2[:, 1], obb2[:, 2], obb2[:, 3], obb2[:, 4], obb2[:, 5]

        # Center distances
        center_dist = torch.sqrt((dx1[:, None] - dx2[None, :]) ** 2 + (dy1[:, None] - dy2[None, :]) ** 2)

        # Orientation differences
        orientation_diff = 1 - torch.abs(cos1[:, None] * cos2[None, :] + sin1[:, None] * sin2[None, :])

        # Dimension differences
        dimension_dist = torch.sqrt((w1[:, None] - w2[None, :]) ** 2 + (l1[:, None] - l2[None, :]) ** 2)

        # Combine metrics
        combined_dist = center_dist + orientation_diff + dimension_dist
        return combined_dist

    def compute_cost_matrix(self, outputs, targets):
        out_bbox = outputs["pred_boxes"].reshape(-1, 6)  # Flatten the bbox outputs
        tgt_bbox = targets["boxes"].reshape(-1, 6)  # Flatten the target bboxes

        # Calculate approximate OBB distance
        cost_bbox = self.obb_distance(out_bbox, tgt_bbox)

        # Classification cost using softmax probabilities
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        cost_class = -out_prob[:, targets["labels"].long()]

        # Combine costs with respective weights
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        return C

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        print("In DETRLoss")
        print(outputs["pred_logits"].shape)
        print(targets["labels"].shape)
        # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]


        # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_bbox = torch.cat([v["boxes"] for v in targets])


        # print(targets.keys())
        tgt_ids = targets["labels"].long()
        tgt_bbox = targets["boxes"].float()  # 原本是double


        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # cost_class = -out_prob[:, tgt_ids]


        # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # 欧式距离
        # cost_bbox = F.smooth_l1_loss(out_bbox, tgt_bbox, reduction='none').sum(dim=1)


        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))  # 应该如何修改

        # Final cost matrix
        # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        C = self.compute_cost_matrix(outputs, targets)
        C = C.view(bs, num_queries, -1).cpu()

        # sizes = [len(v["boxes"]) for v in targets]
        sizes = targets["boxes"].shape[0]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args["set_cost_class"], cost_bbox=args["set_cost_bbox"], cost_giou=args["set_cost_giou"])


