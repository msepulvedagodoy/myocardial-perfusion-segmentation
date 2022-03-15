import torchvision
import random

class RotationTransform(object):

    def __init__(self):
        super().__init__()

    def __call__(self, sample):

        img, mark = sample['img'], sample['mark']

        if random.random() > 0.5:
            angle = random.randint(-360, 360)
            img = torchvision.transforms.functional.rotate(img, angle)
            mark = torchvision.transforms.functional.rotate(mark, angle)

        return img, mark 

class VflipTransform(object):

    def __init__(self) -> None:
        super().__init__()

    
    def __call__(self, sample):

        img, mark = sample['img'], sample['mark']

        if random.random() > 0.5:

            img = torchvision.transforms.functional.vflip(img)
            mark = torchvision.transforms.functional.vflip(mark)

        return img, mark

class HflipTransform(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample):
        
        img, mark = sample['img'], sample['mark']

        if random.random() > 0.5:

            img = torchvision.transforms.functional.hflip(img)
            mark = torchvision.transforms.functional.hflip(mark)
        
        return img, mark

class AdjustbrightnessTransform(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample):

        img, mark = sample['img'], sample['mark']

        if random.random() > 0.5:
            alpha = random.uniform(0.5, 2)
            img = torchvision.transforms.functional.adjust_brightness(img, alpha)

        return img, mark


class AdjustcontrastTransform(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample):

        img, mark = sample['img'], sample['mark']

        if random.random() > 0.5:
            alpha = random.uniform(0.5, 2)
            img = torchvision.transforms.functional.adjust_contrast(img, alpha)

        return img, mark


    
        

        
