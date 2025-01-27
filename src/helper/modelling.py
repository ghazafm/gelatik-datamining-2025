import torch
from torchvision.ops import nms

def generate_proposals(rpn_logits, rpn_bboxes, score_thresh=0.5, nms_thresh=0.3, max_proposals=100):
    """
    Generate proposals from the RPN output.

    Parameters:
    - rpn_logits: Tensor of shape [batch_size, num_anchors, H, W]
    - rpn_bboxes: Tensor of shape [batch_size, num_anchors * 4, H, W]
    - score_thresh: Minimum score threshold to consider a proposal
    - nms_thresh: IoU threshold for NMS
    - max_proposals: Maximum number of proposals to keep per image

    Returns:
    - proposals: List of proposals for each image in the batch (bounding boxes)
    """
    batch_size = rpn_logits.size(0)
    all_proposals = []

    for i in range(batch_size):
        # Get logits and bboxes for a single image
        logits = rpn_logits[i]  # Shape: [num_anchors, H, W]
        bboxes = rpn_bboxes[i]  # Shape: [num_anchors * 4, H, W]

        # Reshape logits and bboxes to [num_anchors * H * W] and [num_anchors * 4 * H * W] respectively
        logits = logits.view(-1)  # Flatten logits to 1D: [num_anchors * H * W]
        bboxes = bboxes.view(-1, 4)  # Flatten bboxes to [num_anchors * H * W, 4]

        # Apply sigmoid to logits to get objectness scores
        scores = torch.sigmoid(logits)

        # Filter out boxes with low scores
        high_score_idx = scores > score_thresh
        scores = scores[high_score_idx]
        bboxes = bboxes[high_score_idx]

        if len(scores) == 0:
            continue

        # Apply Non-Maximum Suppression (NMS) to suppress redundant boxes
        keep_idx = nms(bboxes, scores, nms_thresh)

        # Keep only top N proposals (sorted by score)
        keep_idx = keep_idx[:max_proposals]
        bboxes = bboxes[keep_idx]
        
        # Add batch index to proposals
        batch_indices = torch.full((bboxes.size(0), 1), i, dtype=torch.float32, device=bboxes.device)
        proposals = torch.cat([batch_indices, bboxes], dim=1)  # Shape: [num_proposals, 5]

        all_proposals.append(proposals)
    
    if len(all_proposals) == 0:
        return torch.empty((0, 5), dtype=torch.float32, device=rpn_logits.device)
    
    all_proposals = torch.cat(all_proposals, dim=0)
    return all_proposals
