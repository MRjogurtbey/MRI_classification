"""
Focal Loss Implementation for handling class imbalance and hard examples.

Paper: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
        - p_t: predicted probability for the true class
        - alpha_t: weighting factor (class weights)
        - gamma: focusing parameter (default: 2.0)
    
    Args:
        alpha: Class weights tensor (shape: [num_classes])
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits (before softmax)
            targets: (N,) class indices
        
        Returns:
            Focal loss value
        """
        # Get log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Get probabilities
        probs = torch.exp(log_probs)
        
        # Get probability for the true class
        # Shape: (N,)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Calculate focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_loss = -focal_term * log_pt
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Get alpha for each sample based on target class
            # Shape: (N,)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def test_focal_loss():
    """Test Focal Loss implementation"""
    print("Testing Focal Loss...")
    
    # Create dummy data
    batch_size = 8
    num_classes = 4
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test without class weights
    print("\n1. Focal Loss (no weights):")
    fl1 = FocalLoss(gamma=2.0)
    loss1 = fl1(logits, targets)
    print(f"   Loss: {loss1.item():.4f}")
    
    # Test with class weights
    print("\n2. Focal Loss (with class weights):")
    alpha = torch.tensor([1.5, 1.0, 1.0, 1.0])  # Higher weight for class 0
    fl2 = FocalLoss(alpha=alpha, gamma=2.0)
    loss2 = fl2(logits, targets)
    print(f"   Loss: {loss2.item():.4f}")
    
    # Compare with standard CrossEntropyLoss
    print("\n3. Standard CrossEntropyLoss:")
    ce = nn.CrossEntropyLoss()
    loss3 = ce(logits, targets)
    print(f"   Loss: {loss3.item():.4f}")
    
    print("\n[OK] Focal Loss test passed!")


if __name__ == "__main__":
    test_focal_loss()
