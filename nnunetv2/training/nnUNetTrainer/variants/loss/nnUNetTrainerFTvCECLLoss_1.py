import numpy as np
import torch


from nnunetv2.training.loss.compound_losses import FocalTversky_and_CE_and_Soft_cldice_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerCustomLoss(nnUNetTrainer):
    def _build_loss(self):
        focal_tversky_kwargs = {
            'alpha': 0.3, 
            'beta': 0.7, 
            'gamma': 1.33,
            'smooth': 1e-6,
            'apply_nonlin': torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1
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
        
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
