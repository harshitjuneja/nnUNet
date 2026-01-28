import torch
from torch import nn

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn    

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1.0e-6,
                 apply_nonlin: nn.Module = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

        # what non-linearity to apply to the predictions (e.g., softmax)
        self.apply_nonlin = apply_nonlin

    def forward(self, inputs, targets):
        if self.apply_nonlin is not None:
            inputs = self.apply_nonlin(inputs)

        tp, fp, fn, _ = get_tp_fp_fn_tn(inputs, targets.bool())

        # parameters to control the penalty for false positives and false negatives
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # focal parameter to reduce the relative loss for well-classified examples
        # focal_tversky_loss = torch.pow(-torch.log(tversky_index + 1e-7), self.gamma)
        focal_tversky_loss = torch.pow(torch.clamp(1 - tversky_index, min=1e-7), self.gamma)
        # focal_tversky_loss = (1 - tversky_index) ** self.gamma
        return focal_tversky_loss.mean()


if __name__ == "__main__":

    # alpha=0.3, beta=0.7 favors recall (penalizes False Negatives harder)
    criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33, apply_nonlin=nn.Sigmoid())
    
    targets = torch.zeros((1, 2, 4, 4))
    targets[:, :, 1:3, 1:3] = 1.0
    
    logits_perfect = targets * 10.0 - 5.0 
    # High positive for 1s, negative for 0s
    
    # 4. Case B: Heavy False Negatives (Predicting background where there is foreground)
    # This should be penalized heavily by beta=0.7
    logits_fn = torch.full((1, 2, 4, 4), -5.0) 
    
    # 5. Case C: Heavy False Positives (Predicting foreground where there is background)
    # This should be penalized less by alpha=0.3
    logits_fp = torch.full((1, 2, 4, 4), 5.0)
    loss_perfect = criterion(logits_perfect, targets)
    loss_fn = criterion(logits_fn, targets)
    loss_fp = criterion(logits_fp, targets)

    print("--- Focal Tversky Loss Test Results ---")
    print(f"Perfect Match Loss:      {loss_perfect.item():.6f}")
    print(f"False Negative (Gap):    {loss_fn.item():.6f}")
    print(f"False Positive (Over):   {loss_fp.item():.6f}")

    logits_rand = torch.randn((1, 2, 4, 4), requires_grad=True)
    loss_rand = criterion(logits_rand, targets)
    loss_rand.backward()
    
    print(f"\nGradient Flow Check: {logits_rand.grad is not None}")
































# class FocalTverskyLoss(nn.Module):
#     def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0, smooth: float = 1e-6,
#                  apply_nonlin: nn.Module = None):


#         """
#         Focal Tversky Loss as described in https://arxiv.org/abs/1810.07842

#         :param alpha: weight of false positives
#         :param beta: weight of false negatives
#         :param gamma: focal parameter to reduce the relative loss for well-classified examples
#         :param smooth: smoothing factor to avoid division by zero
#         :param apply_nonlin: non-linearity to apply to the predictions (e.g., softmax)
        
#         """

#         super(FocalTverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.smooth = smooth
#         self.apply_nonlin = apply_nonlin

#     def forward(self, inputs, targets):
        
#         if self.apply_nonlin is not None:
#             inputs = self.apply_nonlin(inputs)

#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         tp, fp, fn, _ = get_tp_fp_fn_tn(inputs, targets, axes=None, loss_mask=None, square=False)
#         tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
#         focal_tversky_loss = (1 - tversky_index) ** self.gamma
        
#         return focal_tversky_loss