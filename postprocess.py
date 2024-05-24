import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import label as LAB
from skimage.transform import resize
import SimpleITK as sitk


def continues_region_extract(label, MR, name):
    regions = np.where(label>=1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, neighbors=4, background=0, connectivity=2, return_num=True)

    # FIND max
    max_num = 0
    for i in range(0, n+1):
        if np.sum(L == i) > max_num and i != 0:
            max_num = np.sum(L == i)
            max_label = i

    vol_e = np.where(label == 4, np.ones_like(label), np.zeros_like(label))
    vol_t = np.where(label >= 1, np.ones_like(label), np.zeros_like(label))
    for i in range(1, n+1):
        # STEP 1 delete labels that are too few or close to edges
        if np.sum(L == i) < min(2000, max_num*0.1) or np.where(L==i)[2].max()<=25:
            label = np.where(L==i, np.zeros_like(label), label)
            if np.sum(L == i) >= min(2000, max_num*0.1) and np.where(L==i)[2].max()<=25:
                print("np.where(L==i)[2].max()<=20!!!")
            continue

        # STEP 2 kmeans
        label_i = np.where(L == i, label, np.zeros_like(label))
        vol_e_i = np.where(label_i == 4, label_i, np.zeros_like(label_i))
        vol_t_i = np.where(label_i >= 1, label_i, np.zeros_like(label_i))
        if vol_e.sum()/vol_t.sum()<0.12 and vol_e_i.sum()/vol_t_i.sum()<0.05 and vol_e_i.sum()<450 and (np.sum(label==1)<10000):
            # K-means
            print("K-means for %s, num_class1 %d" % (name, np.sum(label==1)))
            MR_edema = np.where(label_i==2, MR, np.zeros_like(MR))
            MR_edema_ = np.float32(np.reshape(MR_edema, (-1)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            num_clusters = 3
            ret, matrix, center = cv2.kmeans(MR_edema_, num_clusters, None, criteria, num_clusters, cv2.KMEANS_RANDOM_CENTERS)
            matrix = np.reshape(matrix, newshape=MR_edema.shape)
            center[np.argmax(center)] = np.min(center)
            index = np.argmax(center)
            matrix_convert = np.where(matrix==index, np.ones_like(matrix), np.zeros_like(matrix))
            label = np.where(matrix_convert==1, 1* np.ones_like(label), label)

    NUM_pixels = 500
    # processing ET
    number_4 = np.sum(label == 4)
    if number_4 < NUM_pixels:
        label = np.where(label == 4, 1, label)
    number_4_after = np.sum(label == 4)
    print('%s, B4_%d, A4_%d, num_class1_%d, num_class2_%d\n' % (name, number_4, number_4_after, np.sum(label==1), np.sum(label==2)))

    return label


MR_folder_path = '/home/BraTS-2018/MICCAI_BraTS_2018_Data_Validation/'
img_folder_path = '/home/Desktop/outputs/'
save_path = '/home/Desktop/outputs_posp/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for root, dirs, files in os.walk(img_folder_path):
    for i in sorted(files):
        # if i[:5] != 'BraTS':
        #     continue

        i_file = img_folder_path + '/' + i
        predNII = nib.load(i_file)
        pred = predNII.get_data()

        mr_file = MR_folder_path + '/' + i[:-7] + '/' + i[:-7] + '_t1ce.nii'
        mrNII = nib.load(mr_file)
        mr = mrNII.get_data()

        pred = continues_region_extract(pred.copy(), mr, i)

        pred = pred.astype(np.int16)

        pred = nib.Nifti1Image(pred, affine=predNII.affine)
        name = save_path + '/' + i
        seg_save_p = os.path.join(name)
        nib.save(pred, seg_save_p)

    pass
