import torch
import torchvision
from sklearn.preprocessing import normalize
from scipy.ndimage.morphology import binary_dilation as bd
from scipy.ndimage.morphology import binary_erosion as be
from scipy import ndimage
import numpy as np

class ROI(object):
    def __init__(self) -> None:
        super().__init__()

    def apply_roi(self, slice_imgs, slice_annot=None, compute_metric=False):

        video_shape = slice_imgs.shape
        original_dim = torch.tensor(video_shape[2:4])

        n_frames = video_shape[0]
        downsample_n = 4
        original_dim_downsampled = original_dim // downsample_n
        original_dim_downsampled = list(original_dim_downsampled.numpy())

        slice_imgs_small = torch.zeros([n_frames, 1, *original_dim_downsampled])
        slice_marks_small = torch.zeros([n_frames, 1, *original_dim_downsampled])

        for t in range(n_frames):
            slice_imgs_small[t] = torchvision.transforms.functional.resize(slice_imgs[t], original_dim_downsampled)

            slice_marks_small[t] = torchvision.transforms.functional.resize(slice_annot[t], original_dim_downsampled)

        pre_processing_results = self.pre_processing(slice_imgs_small)
        cum_diff = pre_processing_results['cum_diff']
        filtered_image = pre_processing_results['filtered_image']

        # Do the Graph-Based Visual Saliency
        square_len, start_x, start_y = self.full_gbvs(cum_diff, filtered_image, slice_marks_small, img_size=original_dim_downsampled)

         # Transform the roi rectangle to high dimensionality.
        square_len = downsample_n * square_len
        start_x = downsample_n * start_x
        start_y = downsample_n * start_y

        # Create roi from square length and starting points.
        roi = torch.zeros_like(slice_imgs)
        roi[:, start_x:start_x + square_len, start_y:start_y + square_len, :] = 1

        # Crop images and labels according the roi_dim.
        slice_imgs = slice_imgs[:, :, start_x: start_x + square_len, start_y: start_y + square_len]
        slice_annot = slice_annot[:, :,start_x: start_x + square_len, start_y: start_y + square_len]

        return slice_imgs, slice_annot

    def gaussian_kernel(self, size, mean, std):
        distr = torch.distributions.normal.Normal(mean, std)
        vals = torch.exp(distr.log_prob(torch.range(start=-size, end=size+1)))
        gauss_kernel = torch.einsum('i,j->ij', vals, vals)
        
        return torch.divide(gauss_kernel, torch.sum(gauss_kernel))

    def pre_processing(self, input):
        m, _ = input.shape[2:]
        cum_diff = torch.var(input, dim=0)
        cum_diff = torch.divide(cum_diff, torch.max(cum_diff))
        mask = torch.greater(cum_diff, 0.05)
        mask_f = mask.float()
        cum_diff_padded = cum_diff * mask_f + (torch.tensor(1)-mask_f) * np.percentile(torch.masked_select(cum_diff, mask), 50)

        gauss_kernel = self.gaussian_kernel(size= m/10, mean=0.0, std=m/100)

        gauss_kernel = gauss_kernel[None, None, :, :]
        cum_diff_padded = cum_diff_padded[None, :, :, :]

        # Convolve the image with a gaussian kernel
        filtered_image = torch.nn.functional.conv2d(cum_diff_padded, gauss_kernel, stride=1, padding='same')
        filtered_image = torch.squeeze(filtered_image)
        out = {'cum_diff': cum_diff,
                'mask': mask_f,
                'cum_diff_padded': torch.squeeze(cum_diff_padded),
                'filtered_image': filtered_image}

        return out

    def exp_factor(self, n, sigma):
            y2, x2 = torch.meshgrid(torch.tensor(range(n)), torch.tensor(range(n)))
            x2v = torch.reshape(x2, (1, -1))
            y2v = torch.reshape(y2, (1, -1))
            dx = torch.sub(torch.t(x2v), x2v)
            dy = torch.sub(torch.t(y2v), y2v)
            return torch.exp(-torch.div((torch.pow(dx, 2) + torch.pow(dy, 2)), (2 * (torch.pow(torch.tensor(sigma), 2)))))

    def get_prior(self, size, prior_name, sigma=5):

            if prior_name == 'uniform':
                return torch.reshape(torch.FloatTensor(size, size).uniform_(0, 1), (1, -1))

            elif prior_name == 'gaussian':
                y, x = torch.meshgrid(torch.tensor(range(size)), torch.tensor(range(size)))
                mu = torch.div(size, 2)
                gx = torch.exp(- torch.div(torch.pow(torch.sub(x, mu), 2), torch.pow(2 * torch.tensor(sigma), 2)))
                gy = torch.exp(- torch.div(torch.pow(torch.sub(y, mu), 2), torch.pow(2 * torch.tensor(sigma), 2)))
                return torch.reshape((gx * gy), (1, -1))

    def pix_diff(self, im):
            img_vec = torch.reshape(im, (1,-1))
            return torch.abs( torch.sub(torch.log(torch.t(img_vec)), torch.log(img_vec)))

    def stationary_gbvs(self, img, F, nsteps, prior):
            prior = torch.reshape(prior, (1,-1))
            di = self.pix_diff(img)
            w = torch.tensor(normalize(di * F, norm='l1', axis=1)).float()
            res = torch.matmul(prior, w)
            for i in range(nsteps - 1):
                res = torch.matmul(res, w)
            return res

    def nonstationary_gbvs(self, img, F, n_steps, prior):
            if (type(prior) == bool):
                prior = torch.div(img, torch.max(img))
            prior = torch.reshape(prior, (1,-1))
            for i in range(n_steps):
                if i == 0:
                    di = self.pix_diff(img)
                    w = torch.tensor(normalize(di * F, norm='l1', axis=1)).float()
                    res = torch.matmul(prior, w)
                else:
                    di = self.pix_diff(res)
                    w = torch.tensor(normalize(di * F, norm='l1', axis=1)).float()
                    res = torch.matmul(res, w)
            return res

    def get_bb(self, proc_img, img_shape):

            normalized = torch.reshape(proc_img, img_shape) - torch.min(proc_img)
            normalized = torch.divide(normalized, (torch.max(normalized) - torch.min(normalized)))
            mask = normalized > 0.5

            mask = torch.tensor(bd(mask, iterations=5))
            mask = torch.tensor(be(mask, iterations=5))
            label_im, nb_labels = ndimage.label(mask)
            label_im = torch.tensor(label_im)
            nb_labels = nb_labels

            # Find the largest connected component 
            sizes = torch.tensor(ndimage.sum(mask, label_im, range(nb_labels + 1)))
            mask_size = sizes < torch.max(sizes) * 0.9
            remove_pixel = mask_size[label_im.long()]
            label_im[remove_pixel] = 0
            labels = torch.unique(label_im)
            label_im = torch.tensor(np.searchsorted(labels, label_im))
            slice_x, slice_y = ndimage.find_objects(label_im == 1)[0]
            roi = torch.reshape(proc_img, (img_shape))[slice_x, slice_y]
            return [slice_x, slice_y], roi, mask

    def rectangle_to_square(self, slices, img_size=None):
            img_size = img_size[0]
            slice_x, slice_y = slices
            dx = torch.abs(torch.tensor(slice_x.start - slice_x.stop))
            dy = torch.abs(torch.tensor(slice_y.start - slice_y.stop))
            cm = torch.tensor([slice_x.start + dx//2, slice_y.start + dy//2])

            # Force the square_len to be at most half of the image_size

            square_len = img_size // 2
            start_x = torch.maximum(torch.tensor(0), cm[0] - square_len//2)
            start_y = torch.maximum(torch.tensor(0), cm[1] - square_len//2)
            square_len_x, square_len_y = torch.tensor(square_len), torch.tensor(square_len)

            # If the start of the square is out of the image
            if start_x == 0:
                square_lenx = 2 * cm[0]
            if start_y == 0:
                square_leny = 2 * cm[1]

            square_len = torch.minimum(square_len_x, square_len_y)
            
            # If the end of the square is out of the image
            end_x = start_x + square_len
            end_y = start_y + square_len
            if end_x == img_size:
                square_len_x = torch.abs(end_x - cm[0])
            if end_y == img_size:
                square_len_y = torch.abs(end_y - cm[1])
            square_len = torch.minimum(square_len_x, square_len_y)
            start_x = cm[0] - square_len // 2
            start_y = cm[1] - square_len // 2
            return square_len, start_x, start_y

    def full_gbvs(self, cum_diff_out, filtered_img, marks_mask=None, plot_flag=False, img_size=None, subject_id=None, slice_idx=None, ds_dir=None):
            n = img_size[0]
            exp = self.exp_factor(n, 15)
            g_prior = self.get_prior(n, 'gaussian', sigma=15)
            uprior = self.get_prior(n, 'uniform')
            res_stat = self.stationary_gbvs(filtered_img, exp, 20, uprior)
            res_nonstat = self.nonstationary_gbvs(res_stat, self.exp_factor(n=n, sigma=20), n_steps=3, prior=g_prior)

            slices, _, mask = self.get_bb(res_nonstat, img_size)
            square_len, start_x, start_y = self.rectangle_to_square(slices, img_size)

            return square_len, start_x, start_y

    