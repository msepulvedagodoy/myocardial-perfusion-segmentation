
import torch

class DiceLoss(torch.nn.Module):

   def __init__(self, n_classes=3, smooth = 1.) -> None:
       super(DiceLoss, self).__init__()
       self.n_classes = n_classes
       self.smooth = smooth

   def compute_dice(self, outputs, targets):
     targets = targets.float()
     intersect = torch.sum(torch.mul(outputs, targets))
     y_sum = torch.sum(torch.mul(targets, targets))
     z_sum = torch.sum(torch.mul(outputs, outputs))

     loss = (2. * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
     loss = 1. - loss
     return loss

   def forward(self, outputs, targets):
     
     class_wise_dice = []
     dice_loss = 0.

     for i in range(0, self.n_classes):
       dice = self.compute_dice(outputs[:, i, :, :], targets[:, i, :, :])
       class_wise_dice.append(dice.detach())
       dice_loss += dice 

     losses = {'avg_loss': dice_loss/self.n_classes, 'background_loss': class_wise_dice[0], 
               "epicardium_loss": class_wise_dice[1], "endocardium_loss": class_wise_dice[2]}

     return losses

class DiceLoss2(torch.nn.Module):

  def __init__(self, classes = None, smooth: float = 1., eps: float = 1e-7) -> None:
     super(DiceLoss2, self).__init__()
    
     self.classes = classes
     self.smooth = smooth
     self.eps = eps


  def forward(self, outputs: torch.Tensor, targets: torch.Tensor):

    assert outputs.size(dim=0) == targets.size(dim=0)

    batch_size = outputs.size(dim=0)
    num_classes = outputs.size(dim=1)
    dims = (0, 2)

    y_true = targets.view(batch_size, num_classes, -1)
    y_pred = outputs.view(batch_size, num_classes, -1)

    loss = 1.0 - self.soft_dice_score(y_pred, y_true, dims=dims)

    return loss.mean()

  def soft_dice_score(self, outputs: torch.Tensor, targets: torch.Tensor, dims) -> torch.Tensor:
    
    assert outputs.size() == targets.size()

    if dims is not None:
        intersection = torch.sum(outputs * targets, dim=dims)
        cardinality = torch.sum(outputs + targets, dim=dims)
    else:
        intersection = torch.sum(outputs * targets)
        cardinality = torch.sum(outputs + targets)

    dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)

    return dice_score