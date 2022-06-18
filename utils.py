import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import random

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_single_case(net, img, stride_xy, stride_z, patch_size, num_classes=1):
    image = img.cpu().numpy()
    b, c, d, w, h = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
 

    wl_pad, wr_pad = int(w_pad//2), int(w_pad-w_pad//2)
    hl_pad, hr_pad = int(h_pad//2), int(h_pad-h_pad//2)
    dl_pad, dr_pad = int(d_pad//2), int(d_pad-d_pad//2)
 

    if add_pad:
        image = np.pad(image, [(0, 0), (0, 0),  (dl_pad, dr_pad), (wl_pad,wr_pad), (hl_pad,hr_pad)], mode='constant', constant_values=0)
    bb, cc, dd, ww, hh = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
   
    score_map = np.zeros((bb, num_classes, dd, ww, hh)).astype(np.float32)
    cnt = np.zeros((1, 1, dd, ww, hh)).astype(np.float32)
    out_map = np.zeros((bb, num_classes, dd, ww, hh)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] 
                test_patch = torch.from_numpy(test_patch).cuda()
                

                y1 = net(test_patch, T=10)
                y = torch.softmax(y1, dim=2).mean(0)


                y = y.cpu().data.numpy()
                y1 = y1.cpu().data.numpy() 
                score_map[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                  = score_map[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] + y

                cnt[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                  = cnt[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] + 1

                out_map[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] \
                  = out_map[:, :, zs:zs+patch_size[2], xs:xs+patch_size[0], ys:ys+patch_size[1]] + y1.mean(0)

    score_map = score_map/cnt
    out_map = out_map/cnt 
    if add_pad:
        out_map = out_map[:, :, dl_pad:dl_pad+d, wl_pad:wl_pad+w, hl_pad:hl_pad+h]
        score_map = score_map[:, :, dl_pad:dl_pad+d, wl_pad:wl_pad+w, hl_pad:hl_pad+h]
 
    return torch.from_numpy(out_map).cuda(), torch.from_numpy(score_map).cuda()
