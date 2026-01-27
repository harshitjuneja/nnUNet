import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter=20):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img-img1)
        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, img):
        return self.soft_skel(img)

class Soft_cldice(nn.Module):
    def __init__(self, smooth = 1.0, exclude_background=False):
        super(Soft_cldice, self).__init__()
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


if __name__ == "__main__":

    # We'll use 1 channel (foreground only) for simplicity
    B, C, H, W = 1, 1, 64, 64
    
    criterion = Soft_cldice(exclude_background=False)

    # 2. Create a "Ground Truth" (a simple horizontal line)
    y_true = torch.zeros((B, C, H, W))
    y_true[:, :, 32, 10:54] = 1.0  
    
    # 3. Create a "Perfect Prediction" (should result in low loss)
    y_pred_perfect = y_true.clone()
    
    # 4. Create a "Broken Prediction" (gap in the middle - clDice should hate this)
    y_pred_broken = y_true.clone()
    y_pred_broken[:, :, 32, 30:34] = 0.0 # Add a gap
    
    # 5. Create a "Noisy/Offset Prediction" (shifted up by 1 pixel)
    y_pred_shifted = torch.zeros_like(y_true)
    y_pred_shifted[:, :, 31, 10:54] = 1.0

    loss_perfect = criterion(y_true, y_pred_perfect)
    loss_broken  = criterion(y_true, y_pred_broken)
    loss_shifted = criterion(y_true, y_pred_shifted)

    print("--- clDice Loss Test Results ---")
    print(f"Perfect Match Loss:  {loss_perfect.item():.4f} (Expected: ~0.0)")
    print(f"Broken Line Loss:    {loss_broken.item():.4f} (Expected: Higher than perfect)")
    print(f"Shifted Line Loss:   {loss_shifted.item():.4f} (Expected: Moderate/High)")

    y_pred_rand = torch.rand((B, C, H, W), requires_grad=True)
    
    loss_rand = criterion(y_true, y_pred_rand)
    loss_rand.backward()
    
    print(f"\nGradient check: y_pred_rand.grad is not None = {y_pred_rand.grad is not None}")
