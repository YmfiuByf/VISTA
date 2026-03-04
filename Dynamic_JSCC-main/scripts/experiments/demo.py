# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from options.test_options import TestOptions
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import math
if __name__=='__main__':
    # Extract the options
    opt = TestOptions().parse()

    # Prepare the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=False, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#test images = %d' % dataset_size)

    opt.name = 'C' + str(opt.C_channel) + '_L2_' + str(opt.lambda_L2) + '_re_' + str(opt.lambda_reward) + '_' + opt.select

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.load_networks(200)
    model.eval()

    PSNR_list = []
    SSIM_list = []
    N_channel_list = []
    count_list = [[]]*10
    PSNR_class_list = [[]]*10

    def jscc(input): # [1,3,h,w]
        model.set_input(input.repeat(opt.num_test_channel, 1, 1, 1))
        model.forward()
        fake = model.fake
        hard_mask = model.hard_mask

        N_channel_list.append(hard_mask[0].sum().item())

        # Get the int8 generated images
        img_gen_numpy = fake.detach().cpu().float().numpy()
        img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        img_gen_int8 = img_gen_numpy.astype(np.uint8)

        origin_numpy = input.detach().cpu().float().numpy()
        origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        origin_int8 = origin_numpy.astype(np.uint8)
        return img_gen_int8[0]

    def switch_channel(img,a=0,b=2):
        img = img.copy()
        img[:, :, a] = img[:, :, a] + img[:, :, b]
        img[:, :, b] = img[:, :, a] - img[:, :, b]
        img[:, :, a] = img[:, :, a] - img[:, :, b]
        return img

    result_path = r"D:\pycharmProject\paper\task2\result"
    df = pd.DataFrame(columns=['caption','proposed','Sem+Sem','word+traditional'])
    for file in os.listdir(result_path):
        if '_JSCC' in file: continue
        bits = (len(file)-4)*8
        # print(file)
        img = cv2.imread(os.path.join(result_path,file))
        img = switch_channel(img)
        ori_img = np.array(img)
        img = np.array(img,dtype=np.uint8)
        img = img.astype(np.float32)
        img = (img*2.0)/255.0-1
        img = torch.tensor(img,dtype=torch.float32).to(model.device).permute(2,0,1).unsqueeze(0)
        out = jscc(img)
        out = switch_channel(out)

        out_img = out.copy()
        img = ori_img.copy()
        img = img.reshape((-1,))
        out_img = out_img.reshape((-1,))
        mse = np.sum((img - out_img) ** 2) / len(img)
        psnr = 10 * math.log((255 * 255 / mse), 10)
        print(f'psnr={psnr}',file)

        img = out.astype(np.float32)
        img = (img*2.0)/255.0-1
        img = torch.tensor(img,dtype=torch.float32).to(model.device).permute(2,0,1).unsqueeze(0)
        out = jscc(img)
        out = switch_channel(out)
        out_img = out.copy()
        img = ori_img.copy()
        img = img.reshape((-1,))
        out_img = out_img.reshape((-1,))
        mse = np.sum((img - out_img) ** 2) / len(img)
        psnr = 10 * math.log((255 * 255 / mse), 10)
        print(f'psnr={psnr}', file)

        bits += 29929
        # cv2.imwrite(os.path.join(result_path,file)[:-4]+f'_JSCC.png',out)
        print(f'image caption:{file[:-4]}, bits: 1.proposed={bits},2.Sem+Sem={29929*2},3.word+traditional={128572+bits-29929}')
        df.loc[len(df.index)] = [file, bits, 29929*2,128572+bits-29929]
    # df.to_csv(r"D:\pycharmProject\paper\task2\result.csv")




