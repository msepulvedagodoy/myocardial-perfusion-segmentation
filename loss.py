from turtle import forward
import torch

class DiceLoss(torch.nn.Module):

  def __init__(self):
    super(DiceLoss, self).__init__()
    self.smooth = 1.0

  def forward(self, y_pred, y_true):
    
    score = 0

    for channel in range(y_true.shape[1]):

      y_true_f = torch.flatten(y_true[:,channel, :, :])
      y_pred_f = torch.flatten(y_pred[:,channel, :, :])

      intersection = torch.dot(y_true_f, y_pred_f)

      score += (2. * intersection + self.smooth)/(torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    return 1. - score/y_true.shape[1]

class DiceLoss2(torch.nn.Module):

  def __init__(self, n_classes) -> None:
      super(DiceLoss2, self).__init__()
      self.n_classes = n_classes

  def _dice_loss(self, score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)

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
      dice = self._dice_loss(inputs[:, i], target[:, i])
      class_wise_dice.append(1.0 - dice.item())
      loss += dice * weight[i]

    return loss / self.n_classes


