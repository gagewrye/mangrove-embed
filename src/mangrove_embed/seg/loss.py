"""
Jaccard (IoU) + CE/BCE loss with optional Active Boundary Loss.

Ported unchanged from UCSD-E4E/ml-mangrove (DroneClassification/models/loss.py: JaccardLoss,
ActiveBoundaryLoss; models/utils.py: kl_div).
"""
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


def kl_div(a,b): # q,p
    return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))


class JaccardLoss(nn.Module):
    """
    Jaccard Loss (IoU Loss) for segmentation
    This loss directly optimizes the Intersection over Union metric
    Combined with Cross-Entropy/BCE loss for better gradient flow
    """
    def __init__(self, num_classes=1, ignore_index=255, smooth=1e-6, 
                 weight: Optional[torch.Tensor] = None, alpha=0.5, boundary_weight=0.0):
        """
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in loss calculation (default: 255)
            smooth: Smoothing factor to avoid division by zero
            weight: Optional class weights for handling imbalance
            alpha: Weight between CE and Jaccard loss (0.5 = equal weight. Higher alpha = more CE focus)
            boundary_weight: Weight for Active Boundary Loss component (0.0 = no boundary loss)
        """
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.alpha = alpha
        
        if num_classes == 1:
            # Binary segmentation
            if weight is not None:
                if weight.shape[0] == 2:
                    weight = weight[1].float()
                elif weight.shape[0] > 2:
                    print(f"Warning: weight shape of {weight.shape} is invalid for binary classification, ignoring weight")
                    weight = None
            self.ce = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
            # store processed weight for later IoU/Jaccard computations
            self.iou_weight = torch.sqrt(weight).to(weight.device) if isinstance(weight, torch.Tensor) else None
        else:
            # Multi-class segmentation
            if weight is not None:
                if weight.shape[0] != num_classes:
                    print(f"Warning: weight shape of {weight.shape} is invalid for {num_classes} classes, ignoring weight")
                    weight = None
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
            self.iou_weight = torch.sqrt(weight).to(weight.device) if isinstance(weight, torch.Tensor) else None

        self.boundary_weight = boundary_weight
        if boundary_weight > 0.0:
            self.boundary_loss = ActiveBoundaryLoss(num_classes=num_classes, ignore_index=ignore_index)
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [B, C, H, W] - model logits
            labels: [B, H, W] or [B, 1, H, W] - ground truth class indices
        
        Returns:
            Combined loss: alpha * CE + (1 - alpha) * Jaccard
        """
        logits = logits.float()
        labels = labels.long()

        num_classes = logits.shape[1]
        # Ensure labels are [B, H, W]
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (labels != self.ignore_index)
        
        # --- Cross-Entropy / BCE Loss ---
        if num_classes > 1:
            ce_loss = self.ce(logits, labels)
            ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + self.smooth)
        else:
            ce_loss = self.ce(logits.squeeze(1), labels.float())
            ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + self.smooth)
        
        # --- Jaccard (IoU) Loss ---
        if num_classes == 1:
            # Binary segmentation
            probs = torch.sigmoid(logits.squeeze(1))
            
            # Apply mask
            probs_masked = probs * valid_mask
            targets_masked = labels.float() * valid_mask
            
            # Flatten
            probs_flat = probs_masked.view(-1)
            targets_flat = targets_masked.view(-1)
            
            # IoU calculation
            intersection = (probs_flat * targets_flat).sum()
            union = probs_flat.sum() + targets_flat.sum() - intersection
            iou = (intersection + self.smooth) / (union + self.smooth)
            
        else:
            # Multi-class segmentation
            probs = F.softmax(logits, dim=1)  # [B, C, H, W]
            
            # One-hot encode labels
            labels_one_hot = F.one_hot(labels.clamp(0, self.num_classes - 1), 
                                       num_classes=self.num_classes)  # [B, H, W, C]
            labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            
            # Apply mask to both predictions and targets
            valid_mask_expanded = valid_mask.unsqueeze(1)  # [B, 1, H, W]
            probs_masked = probs * valid_mask_expanded
            labels_masked = labels_one_hot * valid_mask_expanded
            
            # Compute per-class IoU
            intersection = (probs_masked * labels_masked).sum(dim=(0, 2, 3))
            union = (probs_masked + labels_masked).sum(dim=(0, 2, 3)) - intersection
            iou_per_class = (intersection + self.smooth) / (union + self.smooth)
            
            # Average IoU across classes
            iou = (iou_per_class * self.iou_weight).sum() / (self.iou_weight.sum() + self.smooth) if self.iou_weight is not None else iou_per_class.mean()

        jaccard_loss = 1 - iou
        
        if self.boundary_weight > 0.0:
            boundary_loss_value = self.boundary_loss(logits, labels)
            if boundary_loss_value is not None and boundary_loss_value > 0:
                jaccard_loss = (1-self.boundary_weight) * jaccard_loss + self.boundary_weight * boundary_loss_value
        
        # Combined loss
        return self.alpha * ce_loss + (1 - self.alpha) * jaccard_loss


class ActiveBoundaryLoss(nn.Module):
    """
    Multi-class compatible Active Boundary Loss. It helps sharpen boundaries by computing the KL divergence between them.
    This is especially hepful for thin structures.

    This is a vectorized implementation of the Active Boundary Loss described in:
    "Active Boundary Loss for Semantic Segmentation" https://arxiv.org/abs/2102.02696

    - logits: [B, C, H, W]
    - target: [B, H, W] with values in {0..num_classes-1} or ignore_index
    """
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        max_N_ratio: float = 1/100,
        max_clip_dist: float = 20.0,
        detach_kl: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.max_N_ratio = max_N_ratio
        self.max_clip_dist = max_clip_dist
        self.detach_kl = detach_kl

        # CE over 8 directions
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        # 9 offsets (8 dirs + center)
        self.offsets: torch.Tensor
        self.register_buffer(
            "offsets",
            torch.tensor([
                [ 1, 0],
                [-1, 0],
                [ 0,-1],
                [ 0, 1],
                [-1, 1],
                [ 1, 1],
                [-1,-1],
                [ 1,-1],
                [ 0, 0],  # center (ignored)
            ], dtype=torch.long)
        )


    # ------------------------------------------------------------
    # GT → boundary mask
    # ------------------------------------------------------------
    def gt2boundary(self, target):
        B, H, W = target.shape

        # vertical changes
        vert = torch.zeros_like(target, dtype=torch.bool)
        vert[:, 1:] = target[:, 1:] != target[:, :-1]

        # horizontal changes
        horiz = torch.zeros_like(target, dtype=torch.bool)
        horiz[:, :, 1:] = target[:, :, 1:] != target[:, :, :-1]

        boundary = vert | horiz
        boundary |= (target == self.ignore_index)

        return boundary.unsqueeze(1)     # [B,1,H,W]


    # ------------------------------------------------------------
    # Predicted KL-based boundary
    # ------------------------------------------------------------
    def logits2boundary(self, prob):
        B, C, H, W = prob.shape

        # compute pairwise KL spikes
        kl_ud = kl_div(prob[:, :, 1:, :], prob[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(prob[:, :, :, 1:], prob[:, :, :, :-1]).sum(1, keepdim=True)

        kl_map = F.pad(kl_ud, (0,0,0,1)) + F.pad(kl_lr, (0,1,0,0))

        # adaptive threshold
        eps = 1e-5
        max_N = H * W * self.max_N_ratio

        with torch.no_grad():
            while True:
                mask = kl_map > eps
                if mask.sum() > max_N:
                    eps *= 1.2
                else:
                    break

        # dilate to stabilize edges
        mask = mask.float()
        mask = F.max_pool2d(mask, 3, 1, 1) > 0


        return mask.squeeze(1)      # [B,H,W] bool


    # ------------------------------------------------------------
    # Fast signed distance via repeated 3×3 erosions (DT approx)
    # ------------------------------------------------------------
    def get_dist_maps(self, boundary_mask):
        """
        boundary_mask: [B,1,H,W] or [B,H,W]
        Returns [B,H,W] signed distances (negative inside).
        """

        if boundary_mask.dim() == 4:
            boundary_mask = boundary_mask[:,0]

        B, H, W = boundary_mask.shape
        device = boundary_mask.device

        # distance accumulates iterations
        dist = torch.zeros((B, H, W), device=device)

        # working mask: 1=inside, 0=boundary/holes
        active = (~boundary_mask).float()

        # iterate approx EDT (fixed 20 steps max)
        for i in range(1, 21):
            new_active = F.max_pool2d(active.unsqueeze(1), 3, 1, 1).squeeze(1)
            delta = (active > 0) & (new_active < active)
            if not delta.any():
                break
            active[delta] = 0
            dist[delta] = float(i)

        return dist


    # ------------------------------------------------------------
    # FULL LOSS FORWARD
    # ------------------------------------------------------------
    def forward(self, logits, target):
        B, C, H, W = logits.shape
        device = logits.device

        # resize if needed
        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(logits, target.shape[1:], mode="bilinear", align_corners=True)

        prob = logits.softmax(dim=1)

        gt_bnd  = self.gt2boundary(target)
        pred_bnd = self.logits2boundary(prob)

        # nothing predicted
        if pred_bnd.sum() == 0:
            return None

        # signed distance
        dist = self.get_dist_maps(gt_bnd)   # [B,H,W]

        # gather boundary coords (fix batch=1 issue)
        coords = torch.nonzero(pred_bnd, as_tuple=False)

        if coords.numel() == 0:
            return None

        # coords shape can be [K,2] or [K,3]
        if coords.shape[1] == 2:
            b_idx = torch.zeros(coords.size(0), dtype=torch.long, device=device)
            y_idx = coords[:,0]
            x_idx = coords[:,1]
        elif coords.shape[1] == 3:
            b_idx, y_idx, x_idx = coords.T
        else:
            raise RuntimeError(f"Unexpected coords shape {coords.shape}")

        K = coords.size(0)

        # offsets to neighbors
        off = self.offsets.to(device)    # [9,2]

        oy = (y_idx[:,None] + off[:,0]).clamp(0, H-1)
        ox = (x_idx[:,None] + off[:,1]).clamp(0, W-1)

        # distances for GT direction
        dist_vals = dist[b_idx[:,None], oy, ox]    # [K,9]
        dir_gt = torch.argmin(dist_vals, dim=1)
        valid = (dir_gt != 8)

        if not valid.any():
            return None

        dir_gt = dir_gt[valid]
        oy8 = oy[valid, :8]
        ox8 = ox[valid, :8]
        b_valid = b_idx[valid]
        y_valid = y_idx[valid]
        x_valid = x_idx[valid]

        # KL(center || neighbors)
        prob_hw = prob.permute(0,2,3,1)  # [B,H,W,C]

        center = prob_hw[b_valid, y_valid, x_valid]       # [K,C]
        neigh  = prob_hw[b_valid[:,None], oy8, ox8]       # [K,8,C]

        if self.detach_kl:
            neigh = neigh.detach()

        kl_vals = kl_div(center.unsqueeze(1).expand_as(neigh)+1e-8, neigh).sum(dim=2)  # [K,8]

        # distance weights
        w = -dist[b_valid, y_valid, x_valid]
        w = torch.clamp(w / self.max_clip_dist, 0, 1)

        # CE over 8 directions
        loss = self.criterion(kl_vals, dir_gt)
        loss = (loss * w).mean()

        return loss
