import argparse
import sys
sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from DualNet_SS import DualNet_SS as DualNet
from BraTSDataSet import BraTSEvalDataSet
import os
from math import ceil
import nibabel as nib


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = argparse.ArgumentParser(description="Shared-Specific model for 3D medical image segmentation.")

    parser.add_argument("--data_dir", type=str, default='./datalist/')
    parser.add_argument("--data_list", type=str, default='BraTS20_test.csv',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default='80,160,160',
                        help="Comma-separated string with depth, height and width of sub-volumnes.")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes to predict (ET, WT, TC).")
    parser.add_argument("--restore_from", type=str, default='snapshots/conresnet/your_checkpoint_model.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--weight_std", type=bool, default=True,
                        help="whether to use weight standarization in CONV layers.")

    parser.add_argument("--norm_cfg", type=str, default='IN')  # normalization
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')  # activation
    parser.add_argument("--mode", type=str, default='0,1,2,3')
    return parser.parse_args()


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    padded_img = np.pad(img, ((0, 0), (0, 0),(0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(args, net, img_list, tile_size, classes):
    image, image_res = img_list
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))
    count_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                y1 = int(row * strideHW)
                x1 = int(col * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                y2 = min(y1 + tile_size[1], image_size[3])
                x2 = min(x1 + tile_size[2], image_size[4])
                d1 = max(int(d2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]
                padded_img = pad_image(img, tile_size)
                padded_img_res = pad_image(img_res, tile_size)
                padded_prediction, _, _, _ = net(torch.from_numpy(padded_img).cuda(), val=True, mode=args.mode)
                padded_prediction = F.sigmoid(padded_prediction)  # calc sigmoid earlier

                padded_prediction = interp(padded_prediction).cpu().data  # interp
                padded_prediction = padded_prediction[0].cpu()
                prediction = padded_prediction[0:img.shape[2],0:img.shape[3], 0:img.shape[4], :]
                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    # full_probs = torch.sigmoid(full_probs)  # calc sigmoid later

    full_probs = full_probs.numpy().transpose(1,2,3,0)
    return full_probs


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) +1

    dice = 2*num / den

    return dice.mean()


def main():

    args = get_arguments()

    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    d, h, w = map(int, args.input_size.split(','))

    input_size = (d, h, w)

    model = DualNet(args=args, norm_cfg=args.norm_cfg, activation_cfg=args.activation_cfg,
                    num_classes=args.num_classes, weight_std=args.weight_std, self_att=True, cross_att=False)
    model = nn.DataParallel(model)

    print('loading from checkpoint: {}'.format(args.restore_from))
    if os.path.exists(args.restore_from):
        # model.load_state_dict(torch.load(args.restore_from, map_location=torch.device('cpu')))
        checkpoint = torch.load(args.restore_from)
        model = checkpoint['model']
        trained_iters = checkpoint['iter']
        print("Loaded model trained for", trained_iters, "iters")
    else:
        print('File not exists in the reload path: {}'.format(args.restore_from))
        exit(0)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(
        BraTSEvalDataSet(args.data_dir, args.data_list),
        batch_size=1, shuffle=False, pin_memory=True)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    dice_ET = 0
    dice_WT = 0
    dice_TC = 0

    for index, batch in enumerate(testloader):
        # image, image_res, label, size, name, affine = batch
        image, image_res, size, name, affine = batch
        size = size[0].numpy()
        affine = affine[0].numpy()
        # name[0]=name[0].replace("Brats17", "Brats18")
        with torch.no_grad():
            output = predict_sliding(args, model, [image.numpy(),image_res.numpy()], input_size, args.num_classes)

        seg_pred_3class = np.asarray(np.around(output), dtype=np.uint8)

        seg_pred_ET = seg_pred_3class[:, :, :, 0]
        seg_pred_WT = seg_pred_3class[:, :, :, 1]
        seg_pred_TC = seg_pred_3class[:, :, :, 2]
        seg_pred = np.zeros_like(seg_pred_ET)
        seg_pred = np.where(seg_pred_WT == 1, 2, seg_pred)
        seg_pred = np.where(seg_pred_TC == 1, 1, seg_pred)
        seg_pred = np.where(seg_pred_ET == 1, 4, seg_pred)

        # seg_gt = np.asarray(label[0].numpy()[:size[0], :size[1], :size[2]], dtype=np.int)
        # seg_gt_ET = seg_gt[0, :, :, :]
        # seg_gt_WT = seg_gt[1, :, :, :]
        # seg_gt_TC = seg_gt[2, :, :, :]
        #
        # dice_ET_i = dice_score(seg_pred_ET[None, :, :, :], seg_gt_ET[None, :, :, :])
        # dice_WT_i = dice_score(seg_pred_WT[None, :, :, :], seg_gt_WT[None, :, :, :])
        # dice_TC_i = dice_score(seg_pred_TC[None, :, :, :], seg_gt_TC[None, :, :, :])

        # print('Processing {}: Dice_ET = {:.4}, Dice_WT = {:.4}, Dice_TC = {:.4}'.format(name, dice_ET_i, dice_WT_i, dice_TC_i))
        #
        # dice_ET += dice_ET_i
        # dice_WT += dice_WT_i
        # dice_TC += dice_TC_i

        seg_pred = seg_pred.transpose((1,2,0))

        seg_pred = seg_pred.astype(np.int16)

        seg_pred = nib.Nifti1Image(seg_pred, affine=affine)
        seg_save_p = os.path.join('outputs/%s.nii.gz' % (name[0]))
        nib.save(seg_pred, seg_save_p)

    dice_ET_avg = dice_ET / (index + 1)
    dice_WT_avg = dice_WT / (index + 1)
    dice_TC_avg = dice_TC / (index + 1)

    print('Average score: Dice_ET = {:.4}, Dice_WT = {:.4}, Dice_TC = {:.4}'.format(dice_ET_avg, dice_WT_avg, dice_TC_avg))


if __name__ == '__main__':
    main()
