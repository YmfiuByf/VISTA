# img0_path = "D:\\pycharmProject\\video semantic communication\\ball\\balla_1.png"
# img1_path = "D:\\pycharmProject\\video semantic communication\\ball\\ballb_1.png"

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from moviepy.editor import VideoFileClip
from tqdm import tqdm,trange
from savelist import savelist
from PIL import Image
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


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

total_time = 0
ni = 0
def main(k,snr,flag,video):
    # if video == 'street1':
    #     file_name = 'final_street2'
    # else:
    #     file_name = 'final_street'
    assert video in ['street1','street2','street3','street4']
    file_name = 'final_' + video
    # input_path = r"D:\pycharmProject\paper\videos\final_street\street1_resized_beh1_JSCC_SNR=.avi"
    # if flag==1:
    #     input_path = r"D:\pycharmProject\paper\videos\final_street\street1_resized_JSCC_SNR=.avi"
    # input_path = input_path[:-4] + f'{snr}.avi'
    # output_path = input_path[:-4] + f'_int{k - 1}.avi'
    # if flag==1:
    #     output_path = input_path[:-6] + f'{snr}_int{k - 1}.avi'
    input_path = f"D:\\pycharmProject\\paper\\videos\\{file_name}\\street1_resized_beh1_JSCC_SNR={snr}.avi"
    output_path = f"D:\\pycharmProject\\paper\\videos\\{file_name}\\street1_resized_beh1_JSCC_SNR={snr}_int{k-1}.avi"
    if flag ==1:
        input_path = f"D:\\pycharmProject\\paper\\videos\\{file_name}\\street1_resized_JSCC_SNR={snr}.avi"
        output_path = f"D:\\pycharmProject\\paper\\videos\\{file_name}\\street1_resized_JSCC_SNR={snr}_int{k-1}.avi"
    print(input_path,output_path)
    clip = VideoFileClip(input_path)
    video = cv2.VideoCapture(input_path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    print(clip.duration)  # seconds
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

    parser.add_argument('--resume',
                        default="D:\\pycharmProject\\video semantic communication\\VFIformer-main\\VFIformer-main\\pretrained_models\\pretrained_VFIformer\\net_220.pth",
                        type=str)
    parser.add_argument('--resume_flownet', default='', type=str)
    parser.add_argument('--save_folder', default='D:\\pycharmProject\\video semantic communication\\test_results_demo',
                        type=str)

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
    # device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
    device = torch.device('cpu')
    args.device = device
    net = define_G(args)
    net = load_networks(net, args)
    net.eval()

    def imageBGR2RGB(image):
        [B, G, R] = np.split(image, indices_or_sections=3, axis=2)
        ret = np.concatenate([R, G], axis=2)
        ret = np.concatenate([ret, B], axis=2)
        return ret

    ## load data
    def interpolation(front, back, k):
        global total_time
        global ni
        image_list = []
        if k == 1:
            image_list.append(front)
            image_list.append(back)
            return image_list
        elif k == 2:
            mid = one_interpolation(front, back) * 255
            # cv2.imshow('0',mid)
            # cv2.waitKey()
            # mid = imageBGR2RGB(mid)
            image_list.append(front)
            image_list.append(mid)
            image_list.append(back)
            # cv2.imshow('0', image_list[1])
            # cv2.waitKey()
            return image_list
        else:
            mid = one_interpolation(front, back) * 255
            # mid = imageBGR2RGB(mid)
            list1 = interpolation(front, mid, k / 2)
            list2 = interpolation(mid, back, k / 2)
            list2.pop(0)
            return list1 + list2

    def one_interpolation(img0, img1,flow=None):
        divisor = 64
        multi = 3
        global total_time, ni
        t = time.time()
        # c, h, w =img0.shape
        h, w, c = img0.shape
        if h % divisor != 0 or w % divisor != 0:
            h_new = math.ceil(h / divisor) * divisor
            w_new = math.ceil(w / divisor) * divisor
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            # print(type(img0),type(img1),img1.shape)
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                      value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0

        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)

        with torch.no_grad():
            output, _ = net(img0, img1, flow)
            h, w = output.size()[2:]
            output = output[:, :, pad_t:h - pad_d, pad_l:w - pad_r]

        imt = output[0]  # .flip(dims=(0,)).clamp(0., 1.)
        # print(f'imt_shape={imt.size()}')
        t = time.time() - t
        if ni != 0:
            total_time += t
        ni += 1
        # print(t)
        return imt.permute(1, 2, 0).cpu().numpy()

    def one_interpolation2(img0, img1,flow):
        divisor = 64
        multi = 3

        # c, h, w =img0.shape
        h, w, c = img0.shape
        if h % divisor != 0 or w % divisor != 0:
            h_new = math.ceil(h / divisor) * divisor
            w_new = math.ceil(w / divisor) * divisor
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            # print(type(img0),type(img1),img1.shape)
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                      value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0

        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)

        with torch.no_grad():
            output, _ = net(img0, img1, flow)
            h, w = output.size()[2:]
            output = output[:, :, pad_t:h - pad_d, pad_l:w - pad_r]

        imt = output[0]  # .flip(dims=(0,)).clamp(0., 1.)
        # print(f'imt_shape={imt.size()}')
        return imt.permute(1, 2, 0).cpu().numpy()

    def get_flow(img0,img1):
        divisor = 64
        multi = 3
        h, w, c = img0.shape
        if h % divisor != 0 or w % divisor != 0:
            h_new = math.ceil(h / divisor) * divisor
            w_new = math.ceil(w / divisor) * divisor
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            # print(type(img0),type(img1),img1.shape)
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                      value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
                pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0
        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        imgs = torch.cat((img0,img1),1)
        flow,flow_list = net.flownet(imgs)
        flow, c0, c1 = net.refinenet(img0, img1, flow)
        return flow

    def int_video(video,out,k):
        cnt = 1
        success, frame = video.read()
        out.write(frame)
        front, back = frame.copy(),frame.copy()
        for i in trange(20):
            success,back = video.read()
            if not success:
                break
            if cnt==k:

                # front = cv2.resize(front,(55,31))
                # back = cv2.resize(back,(55,31))

                img_list = interpolation(front,back,k)
                front = back
                cnt = 1
                for num in range(1,len(img_list)):
                    tmp_path = r"D:\pycharmProject\video semantic communication\test_results_demo\balla_1_inter.png"
                    cv2.imwrite(tmp_path, img_list[num])
                    tmp_img = cv2.imread(tmp_path)
                    out.write(tmp_img)
            else:
                cnt+=1
        return

    int_video(video,out,k)
    if ni !=0:
        print(f'total_time={total_time},time per interpolation={total_time/(ni-1)},num of interpolation={ni-1}')

def get_time(k,size=(320,184)):
    video = cv2.VideoCapture(r"D:\pycharmProject\paper\videos\street0_resized.avi")
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

    parser.add_argument('--resume',
                        default="D:\\pycharmProject\\video semantic communication\\VFIformer-main\\VFIformer-main\\pretrained_models\\pretrained_VFIformer\\net_220.pth",
                        type=str)
    parser.add_argument('--resume_flownet', default='', type=str)
    parser.add_argument('--save_folder',
                        default='D:\\pycharmProject\\video semantic communication\\test_results_demo',
                        type=str)

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

    cudnn.benchmark = True
    ## save paths
    save_path = args.save_folder

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## load model
    # device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
    device = torch.device('cpu')
    args.device = device
    net = define_G(args)
    net = load_networks(net, args)
    net.eval()

    def imageBGR2RGB(image):
        [B, G, R] = np.split(image, indices_or_sections=3, axis=2)
        ret = np.concatenate([R, G], axis=2)
        ret = np.concatenate([ret, B], axis=2)
        return ret

    ## load data
    def interpolation(front, back, k):
        global total_time
        global ni
        image_list = []
        if k == 1:
            image_list.append(front)
            image_list.append(back)
            return image_list
        elif k == 2:
            mid = one_interpolation(front, back) * 255
            # cv2.imshow('0',mid)
            # cv2.waitKey()
            # mid = imageBGR2RGB(mid)
            image_list.append(front)
            image_list.append(mid)
            image_list.append(back)
            # cv2.imshow('0', image_list[1])
            # cv2.waitKey()
            return image_list
        else:
            mid = one_interpolation(front, back) * 255
            # mid = imageBGR2RGB(mid)
            list1 = interpolation(front, mid, k / 2)
            list2 = interpolation(mid, back, k / 2)
            list2.pop(0)
            return list1 + list2

    def one_interpolation(img0, img1, flow=None):
        divisor = 64
        multi = 3
        global total_time, ni
        t = time.time()
        # c, h, w =img0.shape
        h, w, c = img0.shape
        if h % divisor != 0 or w % divisor != 0:
            h_new = math.ceil(h / divisor) * divisor
            w_new = math.ceil(w / divisor) * divisor
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            # print(type(img0),type(img1),img1.shape)
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                      value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0

        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)

        with torch.no_grad():
            output, _ = net(img0, img1, flow)
            h, w = output.size()[2:]
            output = output[:, :, pad_t:h - pad_d, pad_l:w - pad_r]

        imt = output[0]  # .flip(dims=(0,)).clamp(0., 1.)
        # print(f'imt_shape={imt.size()}')
        t = time.time() - t
        if ni!=0:
            total_time += t
        ni += 1
        # print(t)
        return imt.permute(1, 2, 0).cpu().numpy()

    def one_interpolation2(img0, img1, flow):
        divisor = 64
        multi = 3

        # c, h, w =img0.shape
        h, w, c = img0.shape
        if h % divisor != 0 or w % divisor != 0:
            h_new = math.ceil(h / divisor) * divisor
            w_new = math.ceil(w / divisor) * divisor
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            # print(type(img0),type(img1),img1.shape)
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                      value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0

        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)

        with torch.no_grad():
            output, _ = net(img0, img1, flow)
            h, w = output.size()[2:]
            output = output[:, :, pad_t:h - pad_d, pad_l:w - pad_r]

        imt = output[0]  # .flip(dims=(0,)).clamp(0., 1.)
        # print(f'imt_shape={imt.size()}')
        return imt.permute(1, 2, 0).cpu().numpy()

    def get_flow(img0, img1):
        divisor = 64
        multi = 3
        h, w, c = img0.shape
        if h % divisor != 0 or w % divisor != 0:
            h_new = math.ceil(h / divisor) * divisor
            w_new = math.ceil(w / divisor) * divisor
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            # print(type(img0),type(img1),img1.shape)
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                      value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0
        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        imgs = torch.cat((img0, img1), 1)
        flow, flow_list = net.flownet(imgs)
        flow, c0, c1 = net.refinenet(img0, img1, flow)
        return flow

    def int_video(video, out, k):
        cnt = 1
        success, frame = video.read()
        front, back = frame.copy(), frame.copy()
        for i in trange(20):
            success, back = video.read()
            if not success:
                break
            if cnt == k:

                front = cv2.resize(front,size)
                back = cv2.resize(back,size)

                img_list = interpolation(front, back, k)
                front = back
                cnt = 1
                for num in range(1, len(img_list)):
                    tmp_path = r"D:\pycharmProject\video semantic communication\test_results_demo\balla_1_inter.png"
                    cv2.imwrite(tmp_path, img_list[num])
                    tmp_img = cv2.imread(tmp_path)
            else:
                cnt += 1
        return
    out = 0
    int_video(video, out, k)
    if ni != 0:
        print(f'total_time={total_time},time per interpolation={total_time / (ni-1)},num of interpolation={ni-1}')


if __name__ == '__main__':
    compute_time= True

    if compute_time:
        get_time(2,(140,81))
    else:
        for flag in [1]:
            for k in [4]: # [1,2,4]:
                for snr in [-9]:#[-9,-6,-3,0,3,6,9,12]:
                    ni = 0
                    main(k,snr,flag,video='street4')


