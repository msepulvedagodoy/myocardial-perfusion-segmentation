import torch
import os
import torchvision
import cv2

class dataset_perfusion(torch.utils.data.Dataset):

    def __init__(self, root, inference = False, transforms = torchvision.transforms.Compose([torchvision.transforms.AutoAugment(),torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor()])) -> None:
        super().__init__()
        self.root = root
        self.inference = inference
        self.transforms = transforms

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
            mark = self.transforms(mark)

        return img, mark

    def __len__(self):
        return len(self.img_dict)