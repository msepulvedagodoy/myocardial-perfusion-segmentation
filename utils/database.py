import torch
import os
import torchvision

class dataset_perfusion(torch.utils.data.Dataset):

    def __init__(self, root, inference = False, normalize = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.5,), (0.5,))])) -> None:
        super().__init__()
        self.root = root
        self.inference = inference
        self.normalize = normalize

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

        img = torchvision.io.read_image(path_img, torchvision.io.ImageReadMode.GRAY).float()
        mark = torchvision.io.read_image(path_mark, torchvision.io.ImageReadMode.GRAY).float()

        if self.normalize:
            img = self.normalize(img)

        return img, mark

    def __len__(self):
        return len(self.img_dict)

def loader(dataset, batch_size, num_workers=4, shuffle=True):

    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=True)

    return input_loader