import os
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision.transforms import Pad
import torchvision.transforms.functional as TF
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def aug_pad(self, aug_0, inp_img, tar_img_enh):
        if aug_0 == 1:  # left - top
            # pad
            h_pad = max(self.ps[0] - inp_img.shape[1], 0)
            w_pad = max(self.ps[1] - inp_img.shape[2], 0)
            pad = Pad((w_pad, h_pad, 0, 0))
            inp_img = pad(inp_img)
            tar_img_enh = pad(tar_img_enh)
        elif aug_0 == 2:
            # pad
            h_pad = max(self.ps[0] - inp_img.shape[1], 0)
            w_pad = max(self.ps[1] - inp_img.shape[2], 0)
            pad = Pad((0, h_pad, w_pad, 0))
            inp_img = pad(inp_img)
            tar_img_enh = pad(tar_img_enh)
        elif aug_0 == 3:
            # pad
            h_pad = max(self.ps[0] - inp_img.shape[1], 0)
            w_pad = max(self.ps[1] - inp_img.shape[2], 0)
            pad = Pad((w_pad, 0, 0, h_pad))
            inp_img = pad(inp_img)
            tar_img_enh = pad(tar_img_enh)
        elif aug_0 == 4:
            # pad
            h_pad = max(self.ps[0] - inp_img.shape[1], 0)
            w_pad = max(self.ps[1] - inp_img.shape[2], 0)
            pad = Pad((0, 0, w_pad, h_pad))
            inp_img = pad(inp_img)
            tar_img_enh = pad(tar_img_enh)
        return inp_img,tar_img_enh

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img_enh = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img_enh = TF.to_tensor(tar_img_enh)

        aug_0 = random.randint(0,1) if inp_img.shape == tar_img_enh.shape else 0
        if aug_0 == 0:
            # resize
            inp_img = TF.resize(inp_img, (self.ps[0], self.ps[1]))
            tar_img_enh = TF.resize(tar_img_enh, (self.ps[0], self.ps[1]))
        elif inp_img.shape[1] < self.ps[0] or inp_img.shape[2] < self.ps[1]:
            inp_img, tar_img_enh = self.aug_pad(aug_0, inp_img, tar_img_enh) # pad
            # randomcrop window
            h_rand = random.randint(0, tar_img_enh.shape[1] - self.ps[0])
            w_rand = random.randint(0, tar_img_enh.shape[2] - self.ps[1])
            inp_img = inp_img[:, h_rand :h_rand + self.ps[0], w_rand :w_rand + self.ps[1]]
            tar_img_enh = tar_img_enh[:, h_rand :h_rand + self.ps[0] , w_rand :w_rand + self.ps[1]]
        else:
            # randomcrop window
            h_rand = random.randint(0, tar_img_enh.shape[1] - self.ps[0])
            w_rand = random.randint(0, tar_img_enh.shape[2] - self.ps[1])
            inp_img = inp_img[:, h_rand :h_rand + self.ps[0], w_rand :w_rand + self.ps[1]]
            tar_img_enh = tar_img_enh[:, h_rand :h_rand + self.ps[0] , w_rand :w_rand + self.ps[1]]

        aug_1 = random.randint(0, 8)
        # Data Augmentations
        if aug_1 == 1:
            inp_img = inp_img.flip(1)
            tar_img_enh = tar_img_enh.flip(1)
        elif aug_1 == 2:
            inp_img = inp_img.flip(2)
            tar_img_enh = tar_img_enh.flip(2)
        elif aug_1 == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img_enh = torch.rot90(tar_img_enh, dims=(1, 2))
        elif aug_1 == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img_enh = torch.rot90(tar_img_enh, dims=(1, 2), k=2)
        elif aug_1 == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img_enh = torch.rot90(tar_img_enh, dims=(1, 2), k=3)
        elif aug_1 == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img_enh = torch.rot90(tar_img_enh.flip(1), dims=(1, 2))
        elif aug_1 == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img_enh = torch.rot90(tar_img_enh.flip(2), dims=(1, 2))

        return inp_img, tar_img_enh

class DataLoaderTrainSR(Dataset):
    def __init__(self, rgb_dir, img_options=None, SR_scale=2):
        super(DataLoaderTrainSR, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']
        self.scale = SR_scale

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img_SR = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img_SR = TF.to_tensor(tar_img_SR)

        aug_0 = random.randint(0, 2)
        if aug_0 == 1:
            # random crop window
            h_rand = random.randint(0, tar_img_SR.shape[1] - self.ps[0])
            w_rand = random.randint(0, tar_img_SR.shape[2] - self.ps[1])
            inp_img = inp_img[:, h_rand // self.scale:h_rand // self.scale + self.ps[0] // self.scale,
                      w_rand // self.scale:w_rand // self.scale + self.ps[1] // self.scale]
            tar_img_SR = tar_img_SR[:, h_rand:h_rand + self.ps[0], w_rand:w_rand + self.ps[1]]
        else:
            # resize
            inp_img = TF.resize(inp_img, (self.ps[0] // self.scale, self.ps[1] // self.scale))
            tar_img_SR = TF.resize(tar_img_SR, (self.ps[0], self.ps[1]))

        aug_1 = random.randint(0, 8)
        if aug_1 == 1:
            inp_img = inp_img.flip(1)
            tar_img_SR = tar_img_SR.flip(1)
        elif aug_1 == 2:
            inp_img = inp_img.flip(2)
            tar_img_SR = tar_img_SR.flip(2)
        elif aug_1 == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img_SR = torch.rot90(tar_img_SR, dims=(1, 2))
        elif aug_1 == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img_SR = torch.rot90(tar_img_SR, dims=(1, 2), k=2)
        elif aug_1 == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img_SR = torch.rot90(tar_img_SR, dims=(1, 2), k=3)
        elif aug_1 == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img_SR = torch.rot90(tar_img_SR.flip(1), dims=(1, 2))
        elif aug_1 == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img_SR = torch.rot90(tar_img_SR.flip(2), dims=(1, 2))

        return inp_img, tar_img_SR

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img_enh = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img_enh = TF.to_tensor(tar_img_enh)

        inp_img = TF.resize(inp_img, (self.ps[0], self.ps[1]))
        tar_img_enh = TF.resize(tar_img_enh, (self.ps[0], self.ps[1]))

        return inp_img, tar_img_enh

class DataLoaderValSR(Dataset):
    def __init__(self, rgb_dir, img_options=None, SR_scale=2):
        super(DataLoaderValSR, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']
        self.scale = SR_scale

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img_SR = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img_SR = TF.to_tensor(tar_img_SR)

        inp_img = TF.resize(inp_img,(self.ps[0]//self.scale, self.ps[1]//self.scale))
        tar_img_SR = TF.resize(tar_img_SR, (self.ps[0], self.ps[1]))

        return inp_img, tar_img_SR

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img_enh = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img_enh = TF.to_tensor(tar_img_enh)

        inp_img = TF.resize(inp_img, (self.ps[0], self.ps[1]))
        tar_img_enh = TF.resize(tar_img_enh, (self.ps[0], self.ps[1]))

        return inp_img, tar_img_enh

class DataLoaderTestSR(Dataset):
    def __init__(self, rgb_dir, img_options=None, SR_scale=2):
        super(DataLoaderTestSR, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']
        self.scale = SR_scale

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img_SR = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img_SR = TF.to_tensor(tar_img_SR)

        inp_img = TF.resize(inp_img, (self.ps[0] // self.scale, self.ps[1] // self.scale))
        tar_img_SR = TF.resize(tar_img_SR, (self.ps[0], self.ps[1]))

        return inp_img, tar_img_SR


class DataLoaderInf(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]


        inp_img = Image.open(inp_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)

        inp_img = TF.resize(inp_img, (self.ps[0], self.ps[1]))

        return inp_img

class DataLoaderInfSR(Dataset):
    def __init__(self, rgb_dir, img_options=None, SR_scale=2):
        super(DataLoaderTestSR, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']
        self.scale = SR_scale

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        inp_img = Image.open(inp_path).convert('RGB')
        inp_img = TF.to_tensor(inp_img)

        inp_img = TF.resize(inp_img, (self.ps[0] // self.scale, self.ps[1] // self.scale))

        return inp_img
