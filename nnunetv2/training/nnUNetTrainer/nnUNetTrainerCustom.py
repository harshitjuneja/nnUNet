import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import FocalTversky_and_CE_and_Soft_cldice_loss
import torch.nn.functional as F

class nnUNetTrainerCustomLoss(nnUNetTrainer):
    def _build_loss(self):
        # Define the arguments for each sub-loss
        focal_tversky_kwargs = {
            'alpha': 0.3, 
            'beta': 0.7, 
            'gamma': 1.33, 
            'apply_nonlin': torch.sigmoid # or softmax depending on your output
        }
        ce_kwargs = {}
        cldice_kwargs = {}

        loss = FocalTversky_and_CE_and_Soft_cldice_loss(
            focal_tversky_kwargs=focal_tversky_kwargs,
            ce_kwargs=ce_kwargs,
            cldice_kwargs=cldice_kwargs,
            weight_ce=1,
            weight_focal_tversky=1,
            weight_soft_cldice=1
        )

        # nnUNet handles deep supervision (multi-scale outputs)
        # We must wrap the loss so it can be applied to all output scales

        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = torch.tensor([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
            
        return loss
