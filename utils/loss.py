
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
       if i > 0:
        dice_loss += dice 

     losses = {'avg_loss': dice_loss/(self.n_classes-1), 'background_loss': class_wise_dice[0], 
               "myocardium_loss": class_wise_dice[1], "endocardium_loss": class_wise_dice[2]}

     return losses