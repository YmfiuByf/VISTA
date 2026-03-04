# Copyright (c) OpenMMLab. All rights reserved.
import sys
# sys.path.insert(0,r"D:\pycharmProject\paper\mmsegmentation-master")
import numpy as np
import os
# print('Get current working directory : ', os.getcwd()[:-5])
sys.path.insert(0,os.getcwd()[:-5])
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from argparse import ArgumentParser
# from env_cat import *
from tqdm import tqdm,trange
import matplotlib.pyplot as plt

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from PIL import Image


# print('Get current working directory : ', os.getcwd(),os.path.join(os.getcwd(),'figure.png'))
img_path = os.path.join(os.getcwd(),'figure.png')
out_env = os.path.join(os.getcwd(),'out_env.png')
out_beh = os.path.join(os.getcwd(),'out_beh.png')
tmp_img_path = os.path.join(os.getcwd(),'tmp.png')

out_path = os.path.join(os.getcwd(),'out.png')

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default=img_path,help='Image file')

    parser.add_argument('--config', default=os.path.join(os.getcwd()[:-5],"configs\segmenter\segmenter_vit-l_mask_8x1_640x640_160k_ade20k.py"),help='Config file')
    parser.add_argument('--checkpoint', default =os.path.join(os.getcwd()[:-5],"configs\segmenter\segmenter_vit-l_mask_8x1_640x640_160k_ade20k_20220614_024513-4783a347.pth"), help='Checkpoint file')
    #

    parser.add_argument('--out-file', default=out_path, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    def img_seg(model,tmp_img_path):
        result = inference_segmentor(model, tmp_img_path)
        # print(out_path)
        result = show_result_pyplot(
            model,
            tmp_img_path,
            result,
            # get_palette(args.palette),
            None,
            opacity=args.opacity,
            out_file=args.out_file)
        # print(result.items())
        result = result[0]
        img = np.array(cv2.imread(tmp_img_path))
        # result = result.reshape((-1,1))
        a, indices = np.unique(result, return_counts=True)
        # print(indices)
        print(a,indices)
        def find_k_max(x,k):
            l = len(x)
            c = []
            total = sum(x)
            area = 0
            for _ in range(k):
                max = 0
                for j in range(l):
                    if x[j]>max and j not in c:
                        max = x[j]
                        idx = j
                        c.append(idx)
                        area+=x[j]
                        if area>0.8*total:
                            return c
            # print(c)
            return c
            # for _ in range(k):
            #     idx = np.argmax(x)
            #     c.append(idx)
            #     np.delete(x,idx)
            # for i in range(k,0,-1):
            #     for j in range(k,i,-1):
            #         if

        idxs = find_k_max(indices,10)  # 10 person
        # print(a,indices)
        idx1 = result==a[idxs[0]]
        for idx in idxs:
            idx1 += result==a[idx]
        # print(img.shape)
        img0 = img.copy()
        img0[idx1]=[0,0,0]   # behavior
        idx2 = result==a[idx]
        img1 = img.copy()
        img1[idx2]=[0,0,0]  # env
        return img0,img1

    frame = cv2.imread(img_path)
    cv2.imwrite(tmp_img_path, frame)
    beh, env = img_seg(model,tmp_img_path)
    # plt.imshow(beh)
    # plt.show()
    cv2.imwrite(out_env,env)
    cv2.imwrite(out_beh,beh)

if __name__ == '__main__':

    main()
