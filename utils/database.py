
import os
import pydicom
import numpy as np
import torch
import torchvision
import nibabel as nib

'''
Images and annotations should be as follows:

dataset_folder
    |__ FOLDER_IMAGES
        |__ 100
            |__ ...
        |__ 101
            |__ ...
        |__ 102
            |__ ...
        |__ ...
    |__ FOLDER_MARKS
        |__ 100.nii.gz
        |__ 101.nii.gz
        |__ ...
'''

class MyocardialPerfusionDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=True) -> None:
       

        self.root = root
        self.transform = transform

        # create a dictionary for images.
        locations = list(sorted(os.listdir(os.path.join(self.root, 'dicom'))))
        img_dict = []
        for loc in locations:
            marks_ = nib.load(os.path.join(self.root, 'marks', '%s.nii.gz' % loc)).get_fdata()
            for index, img in enumerate(list(sorted(os.listdir(os.path.join(self.root, loc))))):
                if np.max(marks_[:,:,index]) > 0:
                    dict_ = {'patient_id': loc, 'img_dir': os.path.join(self.root, loc, img), 'mark': index}
                    img_dict.append(dict_)
        
        self.img_dict = img_dict


    def __getitem__(self, idx):
        path_img = self.img_dict[idx]['img_dir']
        dicom = pydicom.read_file(path_img)
        mark = nib.load(os.path.join(self.root, 'marks', '%s.nii.gz' % self.img_dict[idx]['patient_id'])).get_fdata()[self.img_dict[idx]['mark']]

    