img0_path = r"D:\pycharmProject\video semantic communication\ball\behavior\beh_balla_1.png"
img1_path = r"D:\pycharmProject\video semantic communication\ball\behavior\beh_ballb_1.png"
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import time
import logging
import math
import glob
import cv2
import argparse
import numpy as np
from torch.nn.parallel import DataParallel, DistributedDataParallel
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as data
from skimage.color import rgb2yuv, yuv2rgb

from utils.util import setup_logger, print_args
from utils.pytorch_msssim import ssim_matlab
from models import modules
from models.modules import define_G


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_networks(network, args, strict=True):
    load_path = args.resume
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path, map_location=torch.device('cpu'))
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    if 'optimizer' or 'scheduler' in args.net_name:
        network.load_state_dict(load_net_clean)
    else:
        network.load_state_dict(load_net_clean, strict=strict)

    return network



def main():
    parser = argparse.ArgumentParser(description='inference for a single sample')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--name', default='demo', type=str)
    parser.add_argument('--phase', default='test', type=str)

    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    ## network setting
    parser.add_argument('--net_name', default='VFIformer', type=str, help='')

    ## dataloader setting
    parser.add_argument('--crop_size', default=192, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--img0_path', type=str, required=True)
    parser.add_argument('--img1_path', type=str, required=True)

    parser.add_argument('--resume', default="D:\\pycharmProject\\video semantic communication\\VFIformer-main\\VFIformer-main\\pretrained_models\\pretrained_VFIformer\\net_220.pth", type=str)
    parser.add_argument('--resume_flownet', default='', type=str)
    parser.add_argument('--save_folder', default=r"D:\pycharmProject\video semantic communication\ball\interpolation", type=str)

    ## setup training environment
    args = parser.parse_args()

    ## setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    ## distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        args.rank = -1
        print('Disabled distributed training.')
    else:
        pass
        # args.dist = True
        # init_dist()
        # args.world_size = torch.distributed.get_world_size()
        # args.rank = torch.distributed.get_rank()


    cudnn.benchmark = True
    ## save paths
    save_path = args.save_folder

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## load model
    device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
    args.device = device
    net = define_G(args)
    net = load_networks(net, args)
    net.eval()

    ## load data
    def imageBGR2RGB(image):
        [B,G,R] = np.split(image,indices_or_sections=3,axis=2)
        ret = np.concatenate([R,G], axis=2)
        ret = np.concatenate([ret,B],axis=2)
        return ret

    divisor = 64
    multi = 3
    video_path = r"D:\pycharmProject\video semantic communication\blackmagic_pocket_cinema_camera_no_color_correction (360p).mp4"
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("D:\\pycharmProject\\video semantic communication\\___.avi", fourcc, fps, size)
    # _,img0_ = video.read()
    # _,_ = video.read()
    # _,img1_ = video.read()

    #img0,img1 = img0_,img1_
    img0_ = cv2.imread(img0_path)
    img1_ = cv2.imread(img1_path)
    # img0_ = cv2.imread(args.img0_path)
    # img1_ = cv2.imread(args.img1_path)
    img0 = img0_
    img1 = img1_
    h, w, c = img0.shape
    if h % divisor != 0 or w % divisor != 0:
        h_new = math.ceil(h / divisor) * divisor
        w_new = math.ceil(w / divisor) * divisor
        pad_t = (h_new - h) // 2
        pad_d = (h_new - h) // 2 + (h_new - h) % 2
        pad_l = (w_new - w) // 2
        pad_r = (w_new - w) // 2 + (w_new - w) % 2
        img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)  # cv2.BORDER_REFLECT
        img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    else:
        pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0

    img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
    img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
    print(f'img0.size={img0.size()}')
    with torch.no_grad():
        output, _ = net(img0, img1, None)
        h, w = output.size()[2:]
        output = output[:, :, pad_t:h-pad_d, pad_l:w-pad_r]
    print(f'output_size = {output.size()}')
    imt = output[0] #.flip(dims=(0,)).clamp(0., 1.)
    #imt = imt.permute(1,2,0)
    # torchvision.utils.save_image(imt, os.path.join(save_path, os.path.basename(args.img0_path).split('.')[0]+'_inter'+'.png'))
    print(f'result saved!,shape = {imt.shape}')
    imt_ = imt.permute(1, 2, 0).cpu().numpy()
    #imt_ = imageBGR2RGB(imt_)
    cv2.imshow('',imt_)
    cv2.waitKey()
    cv2.imshow('',img0_)
    cv2.waitKey()
    cv2.imshow('',img1_)
    cv2.waitKey()
    cv2.imwrite(os.path.join(save_path, os.path.basename(args.img0_path).split('.')[0]+'_inter'+'.png'),imt_*255)
    print(type(img0_), type(imt_), imt_.shape, img0_.shape)
    # img__ = cv2.imread(r"D:\pycharmProject\video semantic communication\test_results_demo\balla_1_inter.png")
    # out.write(img0_)
    # out.write(img__)



if __name__ == '__main__':
    main()


