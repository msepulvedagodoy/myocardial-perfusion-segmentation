from turtle import forward
import torchmetrics
import torch
from medpy import metric


class Metrics(torch.nn.Module):

    def __init__(self, n_classes=3) -> None:
        super().__init__()
        self.classes = n_classes

    def forward(self, outputs, targets):
        
        class_dice = []
        class_iou = []
        class_ahd = []
        
        ahd_mask = torch.nn.functional.one_hot(torch.argmax(outputs, dim=1).long(), num_classes=3).permute((0,3,1,2))
        
        for i in range(1, self.classes):
            dice = torchmetrics.functional.dice(outputs[:,i,:,:], targets[:,i,:,:])
            iou = torchmetrics.functional.jaccard_index(outputs[:,i,:,:], targets[:,i,:,:], num_classes=2)
            ahd = metric.binary.hd95(ahd_mask[:,i,:,:].detach().numpy(), targets[:,i,:,:].detach().numpy())

            class_dice.append(dice.detach())
            class_iou.append(iou.detach())
            class_ahd.append(ahd)
        
        dice_epi = torchmetrics.functional.dice(outputs[:,1,:,:] + outputs[:,2,:,:], targets[:,1,:,:] + targets[:,2,:,:]).detach()
        iou_epi = torchmetrics.functional.jaccard_index(outputs[:,1,:,:] + outputs[:,2,:,:], targets[:,1,:,:] + targets[:,2,:,:], num_classes=2)
        ahd_epi = metric.binary.hd95(ahd_mask[:,1,:,:].detach().numpy() + ahd_mask[:,2,:,:].detach().numpy(), targets[:,1,:,:].detach().numpy() + targets[:,2,:,:].detach().numpy())
    

        metrics = {'dice_myocardium':  class_dice[0], 'dice_epicardium': dice_epi , 'dice_endocardium': class_dice[1],
                    'iou_myocardium': class_iou[0] , 'iou_epicardium': iou_epi, 'iou_endocardium': class_iou[1],
                    'ahd_myocardium': class_ahd[0] , 'ahd_epicardium':  ahd_epi , 'ahd_endocardium': class_ahd[1]}
        
        return metrics