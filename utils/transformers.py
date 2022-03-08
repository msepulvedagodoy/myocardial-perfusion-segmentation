import torch
import numpy as np

class ToTensor(object):

    def __call__(self, sample):

        image = torch.from_numpy(sample)
        image = torch.unsqueeze(image, dim=0)

        return image

class ClipNorm(object):

    def __init__(self, lower=0, upper=99):
      self.lower = lower
      self.upper = upper
    

    def __call__(self, sample):

      lower_clip, upper_clip = np.percentile(sample, (self.lower, self.upper))
      clip_sample = sample.clip(min=lower_clip, max=upper_clip)
      clip_sample = self.rescale_intensity(clip_sample, (0, 255))

      return clip_sample

    def rescale_intensity(self, tensor, out_range):

      current_min = torch.min(tensor)
      current_max = torch.max(tensor)

      tensor = (tensor - current_min) / (current_max - current_min)
      tensor = out_range[0] + tensor * (out_range[1] - out_range[0])

      return tensor

class ZeroPad(object):

    def __init__(self, size=256):
      self.size = size

    def __call__(self, sample):
      if max(sample.size()[1], sample.size()[2]) > self.size:
        raise ValueError("Data can't be padded to a smaller shape.")
      
      h_pad = self.size - sample.size()[1]
      w_pad = self.size - sample.size()[2]

      padding_left = w_pad // 2
      padding_right = w_pad - padding_left

      padding_top = h_pad // 2
      padding_bottom = h_pad - padding_top
 
      padding = (padding_left, padding_right, padding_top, padding_bottom)

      paddet_data = torch.nn.functional.pad(sample, padding)

      return paddet_data  