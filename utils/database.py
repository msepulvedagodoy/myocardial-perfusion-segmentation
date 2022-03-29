import torch
import os
import torchvision
import random
from utils.data_augmentation import *


class dataset_mri(torch.utils.data.Dataset):

    def __init__(self, root, inference = False, transforms=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor()]), 
    augmentation = [RotationTransform(), VflipTransform(), HflipTransform(), AdjustbrightnessTransform(), AdjustcontrastTransform()]) -> None:
        super().__init__()
        self.root = root
        self.inference = inference
        self.transforms = transforms
        self.augmentation = augmentation

        locations = list(sorted(os.listdir(os.path.join(self.root, 'imgs'))))
        img_dict = []
        for loc in locations:
            marks = list(sorted(os.listdir(os.path.join(self.root, 'marks', loc))))
            imgs = list(sorted(os.listdir(os.path.join(self.root, 'imgs', loc))))
            for i in range(len(imgs)):
                dict = {'patient_id': loc, 'img_dir': os.path.join(self.root, 'imgs', loc, imgs[i]), 
                            'mark_dir': os.path.join(self.root, 'marks', loc, marks[i])}
                img_dict.append(dict)

        self.img_dict = img_dict

    def __getitem__(self, idx):

        path_img = self.img_dict[idx]['img_dir']
        path_mark = self.img_dict[idx]['mark_dir']

        img = torchvision.io.read_image(path_img, torchvision.io.ImageReadMode.GRAY)
        mark = torchvision.io.read_image(path_mark, torchvision.io.ImageReadMode.GRAY)

        if self.transforms:
            img = self.transforms(img)

        if self.augmentation:
            augment = torchvision.transforms.Compose([random.choice(self.augmentation)])
            img, mark = augment({'img': img, 'mark': mark})

        mark = torch.nn.functional.one_hot(torch.squeeze(mark, dim=0).long(), num_classes=3).permute((2,0,1))

        return img, mark

    def __len__(self):
        return len(self.img_dict)