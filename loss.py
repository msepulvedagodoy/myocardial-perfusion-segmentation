from turtle import forward
import torch

class DiceLoss(torch.nn.Module):

  def __init__(self, n_classes) -> None:
      super(DiceLoss, self).__init__()
      self.n_classes = n_classes

  def _dice_loss(self, score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(torch.mul(score, target))
    
    y_sum = torch.sum(torch.mul(target, target))
    z_sum = torch.sum(torch.mul(score, score))

    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

  def forward(self, inputs, target, weight=None):
    if weight is None:
      weight = [1] * self.n_classes

    assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

    class_wise_dice = []
    loss = 0.0

    for i in range(0, self.n_classes):
      dice = self._dice_loss(inputs[:, i, :, :], target[:, i, :, :])
      class_wise_dice.append(1.0 - dice.item())
      loss += dice * weight[i]

    losses = {'avg_loss': loss / self.n_classes, 'background_loss': class_wise_dice[0], 
              "epicardium_loss": class_wise_dice[1], "endocardium_loss": class_wise_dice[2]}

    return losses


