import torch
import torch.nn as nn
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import FocalTversky_and_CE_and_Soft_cldice_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

import torch.nn.functional as F

class nnUNetTrainerCustomLoss(nnUNetTrainer):
    def _build_loss(self):
        # Define the arguments for each sub-loss
        focal_tversky_kwargs = {
            'alpha': 0.3, 
            'beta': 0.7, 
            'gamma': 1.33,
            'smooth': 1e-6,
            'apply_nonlin': torch.nn.Softmax(dim=1)
        }

        ce_kwargs = {}
        cldice_kwargs = {}

        loss = FocalTversky_and_CE_and_Soft_cldice_loss(
            focal_tversky_kwargs=focal_tversky_kwargs,
            ce_kwargs=ce_kwargs,
            cldice_kwargs=cldice_kwargs,
            weight_ce=1,
            weight_focal_tversky=3.5,
            weight_soft_cldice=2.5
        )

        # We must wrap the loss so it can be applied to all output scales
        # nnUNet handles deep supervision (multi-scale outputs)
        
        
        # if self.enable_deep_supervision:
        #     deep_supervision_scales = self._get_deep_supervision_scales()
        #     weights = torch.tensor([1 / (2**i) for i in range(len(deep_supervision_scales))])
        #     weights = weights / weights.sum()
        #     loss = DeepSupervisionWrapper(loss, weights)
            
        # return loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
    
    
