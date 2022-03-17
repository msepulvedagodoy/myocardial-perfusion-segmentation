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
