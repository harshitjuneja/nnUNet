import torch
from torch import nn


from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.focal_tversky import FocalTverskyLoss
from nnunetv2.training.loss.centerline_dice import Soft_cldice
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FocalTversky_and_CE_and_Soft_cldice_loss(nn.Module):
    def __init__(self, focal_tversky_kwargs, ce_kwargs, cldice_kwargs, 
                 weight_ce=1, weight_focal_tversky=1, weight_soft_cldice=1, 
                 ignore_label=None):    
        super(FocalTversky_and_CE_and_Soft_cldice_loss, self).__init__()
        
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_focal_tversky = weight_focal_tversky
        self.weight_ce = weight_ce
        self.weight_soft_cldice = weight_soft_cldice
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.focal_tversky = FocalTverskyLoss(**focal_tversky_kwargs)
        self.soft_cldice = Soft_cldice(**cldice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Following nnUNet standard:
        net_output: (b, c, x, y(, z)) - Raw Logits
        target: (b, 1, x, y(, z)) - Label Map (indices)
        """
        if self.ignore_label is not None:
            # Check for the assertion like in DC_and_CE_loss
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target'
            mask = target != self.ignore_label
            # For Dice-based losses, replace ignore_label with background (0)
            # Mask will handle excluding these regions from the calculation
            target_topo = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_topo = target
            mask = None

        # 1. Cross Entropy Loss
        # target[:, 0] removes the channel dimension (b, 1, x, y) -> (b, x, y)
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # 2. Focal Tversky Loss 
        # Needs the non-linearity (Sigmoid/Softmax) applied inside or via kwargs
        f_tversky_loss = self.focal_tversky(net_output, target_topo) \
            if self.weight_focal_tversky != 0 else 0

        # 3. Soft clDice Loss
        # Softmax is applied here before passing to clDice
        pred_softmax = torch.softmax(net_output, dim=1)
        soft_cldice_loss = self.soft_cldice(target_topo.float(), pred_softmax) \
            if self.weight_soft_cldice != 0 else 0

        return (self.weight_ce * ce_loss + 
                self.weight_focal_tversky * f_tversky_loss + 
                self.weight_soft_cldice * soft_cldice_loss)

# class FocalTversky_and_CE_and_Soft_cldice_loss(nn.Module):
    
#     def __init__(self, focal_tversky_kwargs, ce_kwargs, cldice_kwargs, weight_ce=1, weight_focal_tversky=1, weight_soft_cldice=1):    
        
#         super(FocalTversky_and_CE_and_Soft_cldice_loss, self).__init__()
        
#         self.weight_focal_tversky = weight_focal_tversky
#         self.weight_ce = weight_ce
#         self.weight_soft_cldice = weight_soft_cldice

#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)
#         self.focal_tversky = FocalTverskyLoss(**focal_tversky_kwargs)
#         self.soft_cldice = Soft_cldice(**cldice_kwargs)

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):

#         ce_loss = self.ce(net_output, target)
#         focal_tversky_loss = self.focal_tversky(net_output, target)
#         soft_cldice_loss = self.soft_cldice(target, torch.softmax(net_output, dim=1))
#         result = self.weight_ce * ce_loss + self.weight_focal_tversky * focal_tversky_loss + self.weight_soft_cldice * soft_cldice_loss
        
#         return result

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

if __name__ == "__main__":

    num_classes = 2
    focal_tversky_kwargs = {'alpha': 0.3, 'beta': 0.7, 'gamma': 1.33, 'apply_nonlin': None}
    ce_kwargs = {}
    cldice_kwargs = {}
    
    criterion = FocalTversky_and_CE_and_Soft_cldice_loss(
        focal_tversky_kwargs, ce_kwargs, cldice_kwargs,
        weight_ce=1, weight_focal_tversky=1, weight_soft_cldice=1
    )

    # 2. Create Dummy Data [B, C, H, W]

    # Target is a Long tensor of indices for CE
    target = torch.zeros((1, 2, 64, 64))

    target[:, 20:40, 20:40] = 1  # A square in the middle
    
    # net_output is Raw Logits (before softmax)

    # We want to test two cases: A bad pred and a good pred
    bad_pred = torch.randn((1, num_classes, 64, 64), requires_grad=True)
    
    # Good pred: high values where target is 1, low where 0
    good_pred = torch.zeros((1, num_classes, 64, 64), requires_grad=True)

    with torch.no_grad():
        good_pred[:, 1, 20:40, 20:40] = 10.0
        good_pred[:, 0, :20, :20] = 10.0

    loss_bad = criterion(bad_pred, target)
    loss_good = criterion(good_pred, target)

    print("--- Compound Loss Test Results ---")
    print(f"Bad Prediction Loss:  {loss_bad.mean():.4f}")
    print(f"Good Prediction Loss: {loss_good.mean():.4f}")

    if loss_good.mean() < loss_bad.mean():
        print("✅ SUCCESS: Loss decreases as prediction improves.")
    else:
        print("❌ FAILURE: Loss did not decrease. Check your logic.")

    loss_bad.backward()
    if bad_pred.grad is not None:
        print(f"✅ SUCCESS: Gradients are flowing to net_output. Mean grad: {bad_pred.grad.abs().mean().item():.6f}")
    else:
        print("❌ FAILURE: No gradients detected!")


        