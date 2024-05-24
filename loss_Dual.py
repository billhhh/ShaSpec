import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        loss_avg = 1 - dice_score.mean()

        return loss_avg

class DiceLoss4BraTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss4BraTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict %s & target %s shape do not match' % (predict.shape, target.shape)
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.sigmoid(predict)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/(target.shape[1]-1 if self.ignore_index!=None else target.shape[1])


class BCELoss4BraTS(nn.Module):
    def __init__(self, ignore_index=None, **kwargs):
        super(BCELoss4BraTS, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights = None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-7, max=1-1e-7)
            bce = weights[1] * (target * torch.log(output)) + \
                  weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                bce_loss = self.criterion(predict[:, i], target[:, i])
                total_loss += bce_loss

        return total_loss.mean()


class BCELossBoud(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(BCELossBoud, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights = None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-3, max=1-1e-3)
            bce = weights[1] * (target * torch.log(output)) + \
                  weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):

        bs, category, depth, width, heigt = target.shape
        bce_loss = []
        for i in range(predict.shape[1]):
            pred_i = predict[:,i]
            targ_i = target[:,i]
            tt = np.log(depth * width * heigt / (target[:, i].cpu().data.numpy().sum()+1))
            bce_i = self.weighted_BCE_cross_entropy(pred_i, targ_i, weights=[1, tt])
            bce_loss.append(bce_i)

        bce_loss = torch.stack(bce_loss)
        total_loss = bce_loss.mean()
        return total_loss


class DiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target):
        # target = target.cuda()
        # predict = predict.cpu()
        #onehot
        # target = target[:,None,:,:,:]
        # target = make_one_hot(target, predict.shape[1])
        #assert predict.shape == target.shape, 'predict %s & target %s shape do not match' % (predict.shape, target.shape)

        total_loss = []
        predict = F.sigmoid(predict)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == self.num_classes, \
                        'Expect weight shape [{}], get[{}]'.format(self.num_classes, self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss==total_loss]
        # total_loss = torch.stack(list(total_loss.values()))

        return total_loss.sum()/total_loss.shape[0]


class CELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(CELoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # self.criterion = torch.nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'

        total_loss = []
        for i in range(self.num_classes):
            if i != self.ignore_index:
                ce_loss = self.criterion(predict[:, i], target[:, i])
                ce_loss = torch.mean(ce_loss, dim=[1,2,3])

                ce_loss_avg = ce_loss[target[:, i, 0, 0, 0] != -1].sum() / ce_loss[target[:, i, 0, 0, 0] != -1].shape[0]

                total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]


class DomainBCELoss(nn.Module):
    def __init__(self):
        super(DomainBCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = self.criterion(predict, target)
        return total_loss.sum() / total_loss.shape[0]


class KDLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(KDLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, source_logits, source_gt, target_logits, target_gt):

        source_prob = []
        target_prob = []

        temperature = 2.0
        loss = 0.0

        # kd process
        for i in range(self.num_classes):
            eps = 1e-6

            s_mask = source_gt[:,i,:,:,:].unsqueeze(1).repeat_interleave(repeats=self.num_classes, dim=1)
            s_logits_mask_out = source_logits * s_mask
            s_logits_avg = torch.sum(s_logits_mask_out, dim=(0,2,3,4)) / (torch.sum(source_gt[:,i,:,:,:]) + eps)
            s_soft_prob = torch.nn.functional.softmax(s_logits_avg / temperature)
            source_prob.append(s_soft_prob)

            t_mask = target_gt[:, i, :, :, :].unsqueeze(1).repeat_interleave(repeats=self.num_classes, dim=1)
            t_logits_mask_out = target_logits * t_mask
            t_logits_avg = torch.sum(t_logits_mask_out, dim=(0, 2, 3, 4)) / (torch.sum(target_gt[:, i, :, :, :]) + eps)
            t_soft_prob = torch.nn.functional.softmax(t_logits_avg / temperature)
            target_prob.append(t_soft_prob)

            # KL divergence loss
            loss = (torch.sum(s_soft_prob * torch.log(s_soft_prob / t_soft_prob)) + torch.sum(
                t_soft_prob * torch.log(t_soft_prob / s_soft_prob))) / 2.0

        return loss


class DomainClsLoss(nn.Module):
    def __init__(self):
        super(DomainClsLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], 'predict & target shape do not match'
        total_loss = self.criterion(predict, target)
        return total_loss
