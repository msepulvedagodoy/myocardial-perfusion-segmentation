
import torch

class DiceLoss(torch.nn.Module):
  def __init__(self, smooth=1., n_classes=3) -> None:
    super(DiceLoss, self).__init__()
    self.smooth = smooth
    self.n_classes = n_classes

  def compute_dice(self, output, target):
    output_flat = output.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (output_flat*target_flat).sum()

    loss = 1. - ((2. * intersection + self.smooth) /(output_flat.sum() + target_flat.sum() + self.smooth))
    return loss

  def forward(self, output, target):
    
    dice_per_class = []
    dice_sum = 0.
    for i in range(0, self.n_classes):
      dice = self.compute_dice(output[:, i, :, :], target[:, i, :, :])
      dice_per_class.append(dice)
      dice_sum += dice

    losses = {'avg_loss': dice_sum/self.n_classes, 'background_loss': dice_per_class[0], 
              "epicardium_loss": dice_per_class[1], "endocardium_loss": dice_per_class[2]}
    return losses