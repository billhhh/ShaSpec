import os.path as osp
import numpy as np
import random
from torch.utils import data
import nibabel as nib
from skimage.transform import resize
import math

from batchgenerators.transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform


class BraTSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(128, 160, 200), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]

        if not max_iters==None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            if item[0] == 'BraTS_2020_subject_ID':
                continue
            filepath = item[0] + '/' + osp.splitext(osp.basename(item[0]))[0]
            # flair_file = filepath + '_flair.nii.gz'  # BraTS20
            # t1_file = filepath + '_t1.nii.gz'
            # t1ce_file = filepath + '_t1ce.nii.gz'
            # t2_file = filepath + '_t2.nii.gz'
            # label_file = filepath + '_seg.nii.gz'

            flair_file = filepath + '_flair.nii'
            t1_file = filepath + '_t1.nii'
            t1ce_file = filepath + '_t1ce.nii'
            t2_file = filepath + '_t2.nii'
            label_file = filepath + '_seg.nii'
            name = osp.splitext(osp.basename(filepath))[0]
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0,:,:,:] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map

    # locate more specific location in bbx for training
    def locate_bbx(self, label):

        class_num, img_d, img_h, img_w = label.shape

        if random.random() < 0.5:
            selected_class = np.random.choice(class_num + 1)
            class_locs = []
            if selected_class != class_num:
                class_label = label[selected_class]
                class_locs = np.argwhere(class_label > 0)

            if selected_class == class_num or len(class_locs) == 0:
                # if no foreground found, then randomly select
                d0 = random.randint(0, img_d - 0 - self.crop_d)
                h0 = random.randint(15, img_h - 15 - self.crop_h)
                w0 = random.randint(10, img_w - 10 - self.crop_w)
                d1 = d0 + self.crop_d
                h1 = h0 + self.crop_h
                w1 = w0 + self.crop_w
            else:
                selected_voxel = class_locs[np.random.choice(len(class_locs))]
                center_d, center_h, center_w = selected_voxel

                d0 = center_d - self.crop_d // 2
                d1 = center_d + self.crop_d // 2
                h0 = center_h - self.crop_h // 2
                h1 = center_h + self.crop_h // 2
                w0 = center_w - self.crop_w // 2
                w1 = center_w + self.crop_w // 2

                if h0 < 0:
                    delta = h0 - 0
                    h0 = 0
                    h1 = h1 - delta
                if h1 > img_h:
                    delta = h1 - img_h
                    h0 = h0 - delta
                    h1 = img_h
                if w0 < 0:
                    delta = w0 - 0
                    w0 = 0
                    w1 = w1 - delta
                if w1 > img_w:
                    delta = w1 - img_w
                    w0 = w0 - delta
                    w1 = img_w
                if d0 < 0:
                    delta = d0 - 0
                    d0 = 0
                    d1 = d1 - delta
                if d1 > img_d:
                    delta = d1 - img_d
                    d0 = d0 - delta
                    d1 = img_d

        else:
            d0 = random.randint(0, img_d - 0 - self.crop_d)
            h0 = random.randint(15, img_h - 15 - self.crop_h)
            w0 = random.randint(10, img_w - 10 - self.crop_w)
            d1 = d0 + self.crop_d
            h1 = h0 + self.crop_h
            w1 = w0 + self.crop_w

        d0 = np.max([d0, 0])
        d1 = np.min([d1, img_d])
        h0 = np.max([h0, 0])
        h1 = np.min([h1, img_h])
        w0 = np.max([w0, 0])
        w1 = np.min([w1, img_w])

        return [d0, d1, h0, h1, w0, w1]

    # with scale
    def locate_bbx_wScale(self, label):

        # randomly scale imgs
        scale_flag = False
        if self.scale and np.random.uniform() < 0.5:
            scaler = np.random.uniform(0.9, 1.1)
        # if self.scale and np.random.uniform() < 0.2:
        #     scaler = np.random.uniform(0.85, 1.25)
            scale_flag = True
        else:
            scaler = 1
        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        class_num, img_d, img_h, img_w = label.shape

        if random.random() < 0.5:
            selected_class = np.random.choice(class_num + 1)
            class_locs = []
            if selected_class != class_num:
                class_label = label[selected_class]
                class_locs = np.argwhere(class_label > 0)

            if selected_class == class_num or len(class_locs) == 0:
                # if no foreground found, then randomly select
                d0 = random.randint(0, img_d - 0 - scale_d)
                h0 = random.randint(15, img_h - 15 - scale_h)
                w0 = random.randint(10, img_w - 10 - scale_w)
                d1 = d0 + scale_d
                h1 = h0 + scale_h
                w1 = w0 + scale_w
            else:
                selected_voxel = class_locs[np.random.choice(len(class_locs))]
                center_d, center_h, center_w = selected_voxel

                d0 = center_d - scale_d // 2
                d1 = center_d + scale_d // 2
                h0 = center_h - scale_h // 2
                h1 = center_h + scale_h // 2
                w0 = center_w - scale_w // 2
                w1 = center_w + scale_w // 2

                if h0 < 0:
                    delta = h0 - 0
                    h0 = 0
                    h1 = h1 - delta
                if h1 > img_h:
                    delta = h1 - img_h
                    h0 = h0 - delta
                    h1 = img_h
                if w0 < 0:
                    delta = w0 - 0
                    w0 = 0
                    w1 = w1 - delta
                if w1 > img_w:
                    delta = w1 - img_w
                    w0 = w0 - delta
                    w1 = img_w
                if d0 < 0:
                    delta = d0 - 0
                    d0 = 0
                    d1 = d1 - delta
                if d1 > img_d:
                    delta = d1 - img_d
                    d0 = d0 - delta
                    d1 = img_d

        else:
            d0 = random.randint(0, img_d - 0 - scale_d)
            h0 = random.randint(15, img_h - 15 - scale_h)
            w0 = random.randint(10, img_w - 10 - scale_w)
            d1 = d0 + scale_d
            h1 = h0 + scale_h
            w1 = w0 + scale_w

        d0 = np.max([d0, 0])
        d1 = np.min([d1, img_d])
        h0 = np.max([h0, 0])
        h1 = np.min([h1, img_h])
        w0 = np.max([w0, 0])
        w1 = np.min([w1, img_w])

        return [d0, d1, h0, h1, w0, w1], scale_flag

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem_scale__(self, index):  # default for scale
        datafiles = self.files[index]
        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])
        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())
        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        label = labelNII.get_data()
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        if self.scale:
            scaler = np.random.uniform(0.9, 1.1)
            scale_flag = True
        else:
            scaler = 1
            scale_flag = False

        # scale_flag = False
        # if self.scale and np.random.uniform() < 0.3:  # random scale with a prob
        #     # scaler = np.random.uniform(0.85, 1.25)
        #     scaler = np.random.uniform(0.9, 1.1)
        #     scale_flag = True
        # else:
        #     scaler = 1

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        d_off = random.randint(0, img_d - scale_d)
        h_off = random.randint(15, img_h - 15 - scale_h)
        w_off = random.randint(10, img_w - 10 - scale_w)

        image = image[:, h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]
        label = label[h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]

        label = self.id2trainId(label)

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))     # Depth x H x W

        if self.is_mirror:
            randi = np.random.rand(1)
            if randi <= 0.3:
                pass
            elif randi <= 0.4:
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            elif randi <= 0.5:
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            elif randi <= 0.6:
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]
            elif randi <= 0.7:
                image = image[:, :, ::-1, ::-1]
                label = label[:, :, ::-1, ::-1]
            elif randi <= 0.8:
                image = image[:, ::-1, :, ::-1]
                label = label[:, ::-1, :, ::-1]
            elif randi <= 0.9:
                image = image[:, ::-1, ::-1, :]
                label = label[:, ::-1, ::-1, :]
            else:
                image = image[:, ::-1, ::-1, ::-1]
                label = label[:, ::-1, ::-1, ::-1]

        if scale_flag:
            image = resize(image, (4, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0, clip=True, preserve_range=True)
            label = resize(label, (3, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy()

    def __getitem_locateBbx__(self, index):  # for locate bbx
        datafiles = self.files[index]
        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])
        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())
        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        label = labelNII.get_data()
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        label = self.id2trainId(label)
        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        [d0, d1, h0, h1, w0, w1] = self.locate_bbx(label)

        image = image[:, d0: d1, h0: h1, w0: w1]
        label = label[:, d0: d1, h0: h1, w0: w1]

        if self.is_mirror:
            randi = np.random.rand(1)
            if randi <= 0.3:
                pass
            elif randi <= 0.4:
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            elif randi <= 0.5:
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            elif randi <= 0.6:
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]
            elif randi <= 0.7:
                image = image[:, :, ::-1, ::-1]
                label = label[:, :, ::-1, ::-1]
            elif randi <= 0.8:
                image = image[:, ::-1, :, ::-1]
                label = label[:, ::-1, :, ::-1]
            elif randi <= 0.9:
                image = image[:, ::-1, ::-1, :]
                label = label[:, ::-1, ::-1, :]
            else:
                image = image[:, ::-1, ::-1, ::-1]
                label = label[:, ::-1, ::-1, ::-1]

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy()

    def __getitem__(self, index):  # for locate bbx with scale
        datafiles = self.files[index]
        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])
        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())
        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        label = labelNII.get_data()
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        label = self.id2trainId(label)
        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        [d0, d1, h0, h1, w0, w1], scale_flag = self.locate_bbx_wScale(label)

        image = image[:, d0: d1, h0: h1, w0: w1]
        label = label[:, d0: d1, h0: h1, w0: w1]

        if self.is_mirror:
            randi = np.random.rand(1)
            if randi <= 0.3:
                pass
            elif randi <= 0.4:
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            elif randi <= 0.5:
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            elif randi <= 0.6:
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]
            elif randi <= 0.7:
                image = image[:, :, ::-1, ::-1]
                label = label[:, :, ::-1, ::-1]
            elif randi <= 0.8:
                image = image[:, ::-1, :, ::-1]
                label = label[:, ::-1, :, ::-1]
            elif randi <= 0.9:
                image = image[:, ::-1, ::-1, :]
                label = label[:, ::-1, ::-1, :]
            else:
                image = image[:, ::-1, ::-1, ::-1]
                label = label[:, ::-1, ::-1, ::-1]

        if scale_flag:
            image = resize(image, (4, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0, clip=True, preserve_range=True)
            label = resize(label, (3, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy()


class BraTSValDataSet(data.Dataset):
    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        for item in self.img_ids:
            if item[0] == 'BraTS_2020_subject_ID':
                continue
            filepath = item[0] + '/' + osp.splitext(osp.basename(item[0]))[0]
            # flair_file = filepath + '_flair.nii.gz'  # BraTS20
            # t1_file = filepath + '_t1.nii.gz'
            # t1ce_file = filepath + '_t1ce.nii.gz'
            # t2_file = filepath + '_t2.nii.gz'
            # label_file = filepath + '_seg.nii.gz'

            flair_file = filepath + '_flair.nii'
            t1_file = filepath + '_t1.nii'
            t1ce_file = filepath + '_t1ce.nii'
            t2_file = filepath + '_t2.nii'
            label_file = filepath + '_seg.nii'
            name = osp.splitext(osp.basename(filepath))[0]
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0, :, :, :] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map


    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]

        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])

        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())
        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        label = labelNII.get_data()
        name = datafiles["name"]

        label = self.id2trainId(label)

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))     # Depth x H x W
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        size = image.shape[1:]
        affine = labelNII.affine

        # image -> res
        cha, dep, hei, wei = image.shape
        image_copy = np.zeros((cha, dep, hei, wei)).astype(np.float32)
        image_copy[:, 1:, :, :] = image[:, 0:dep - 1, :, :]
        image_res = image - image_copy
        image_res[:, 0, :, :] = 0
        image_res = np.abs(image_res)

        return image.copy(), image_res.copy(), label.copy(), np.array(size), name, affine


class BraTSEvalDataSet(data.Dataset):
    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        for item in self.img_ids:
            if item[0] == 'BraTS_2020_subject_ID':
                continue
            filepath = item[0] + '/' + osp.splitext(osp.basename(item[0]))[0]
            # flair_file = filepath + '_flair.nii.gz'  # BraTS20
            # t1_file = filepath + '_t1.nii.gz'
            # t1ce_file = filepath + '_t1ce.nii.gz'
            # t2_file = filepath + '_t2.nii.gz'
            # label_file = filepath + '_seg.nii.gz'

            flair_file = filepath + '_flair.nii'
            t1_file = filepath + '_t1.nii'
            t1ce_file = filepath + '_t1ce.nii'
            t2_file = filepath + '_t2.nii'
            label_file = filepath + '_seg.nii'
            name = osp.splitext(osp.basename(filepath))[0]
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0, :, :, :] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]

        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])

        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())
        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        name = datafiles["name"]

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        image = image.astype(np.float32)

        size = image.shape[1:]
        affine = t1NII.affine

        # image -> res
        cha, dep, hei, wei = image.shape
        image_copy = np.zeros((cha, dep, hei, wei)).astype(np.float32)
        image_copy[:, 1:, :, :] = image[:, 0:dep - 1, :, :]
        image_res = image - image_copy
        image_res[:, 0, :, :] = 0
        image_res = np.abs(image_res)

        return image.copy(), image_res.copy(), np.array(size), name, affine


def get_train_transform(patch_size):
    tr_transforms = []

    tr_transforms.append(
        SpatialTransform(
            patch_size, patch_center_dist_from_border=[i // 2 for i in patch_size],
            do_elastic_deform=True, alpha=(0., 900.), sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True, scale=(0.85, 1.25),
            border_mode_data='constant', border_cval_data=0,
            order_data=3, border_mode_seg="constant", border_cval_seg=-1,
            order_seg=1,
            random_crop=True,
            p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,
            data_key="image", label_key="label")
    )
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="image"))

    tr_transforms.append(MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def my_collate(batch):
    image, label = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    data_dict = {'image': image, 'label':label}
    tr_transforms = get_train_transform(patch_size=image.shape[2:])
    data_dict = tr_transforms(**data_dict)
    return data_dict
