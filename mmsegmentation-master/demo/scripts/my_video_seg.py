# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.insert(0,r"D:\pycharmProject\paper\mmsegmentation-master")
import numpy as np
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from argparse import ArgumentParser
from env_cat import *
from tqdm import tqdm,trange
import matplotlib.pyplot as plt

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

img_path = r"D:\pycharmProject\video semantic communication\ball\balla_2.png"
out_path = "D:/pycharmProject/video semantic communication/ball/seg"+'/'  + img_path[-11:]
tmp_img_path = r"D:\pycharmProject\video semantic communication\ball\tmp_img.png"
from moviepy.editor import VideoFileClip
from PIL import Image
input_video = r"D:\pycharmProject\paper\videos\day_2.avi"
out_beh = input_video[:-4] + '_beh1.avi'
out_b = input_video[:-4] + '_b1.avi'
out_env = input_video[:-4] + '_env1.avi'
env_cat_path = input_video[:-4] + '_env_cat1.png'
from tqdm import tqdm,trange

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default=img_path,help='Image file')
    parser.add_argument('--config', default=r"D:\pycharmProject\paper\mmsegmentation-master\configs\segmenter\segmenter_vit-l_mask_8x1_640x640_160k_ade20k.py",help='Config file')
    parser.add_argument('--checkpoint', default =r"D:\pycharmProject\paper\mmsegmentation-master\configs\segmenter\segmenter_vit-l_mask_8x1_640x640_160k_ade20k_20220614_024513-4783a347.pth", help='Checkpoint file')
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

    clip = VideoFileClip(input_video)
    print(clip.duration)  # seconds
    video = cv2.VideoCapture(input_video)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    size_out = (640, 360)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter(out_beh, fourcc, fps, size)
    out2 = cv2.VideoWriter(out_env,fourcc,fps,size)
    out3 = cv2.VideoWriter(out_b,fourcc,fps,size)
    num = int(fps * clip.duration)


    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    def img_seg(model,tmp_img_path):
        result = inference_segmentor(model, tmp_img_path)
        # show the results
        # print(out_path)
        result = show_result_pyplot(
            model,
            tmp_img_path,
            result,
            # get_palette(args.palette),
            None,
            opacity=args.opacity,
            out_file=args.out_file)
        img = np.array(cv2.imread(tmp_img_path))
        # result = result.reshape((-1,1))
        a, indices = np.unique(result, return_counts=True)
        # print(indices)
        def find_k_max(x,k):
            l = len(x)
            c = []
            for i in range(k):
                max = 0
                for j in range(l):
                    if x[j]>max and j not in c:
                        max = x[j]
                        idx = j
                c.append(idx)
            return c[-1]
            # for _ in range(k):
            #     idx = np.argmax(x)
            #     c.append(idx)
            #     np.delete(x,idx)
            # for i in range(k,0,-1):
            #     for j in range(k,i,-1):
            #         if

        idx = find_k_max(indices,10)  # 10 person
        # print(a,indices)
        idx1 = result==a[idx]
        # print(img.shape)
        img0 = img.copy()
        img0[idx1]=[0,0,0]   # behavior
        idx2 = result==a[idx]
        img1 = img.copy()
        img1[idx2]=[0,0,0]  # env
        return img0,img1

    success, frame = video.read()
    frame = cv2.resize(frame, size, Image.ANTIALIAS)
    # print(type(frame))
    back = frame
    # out.write(frame)
    flag = True
    envs = []
    num = 100
    for _ in trange(num):
        if not success:
            break
        cv2.imwrite(tmp_img_path,frame)
        beh, env = img_seg(model,tmp_img_path)
        # cv2.imshow('',beh)
        # cv2.waitKey()
        if flag:
            out1.write(beh)
        out3.write(beh)
        out2.write(env)
        envs.append(env)
        flag = not flag
        success, frame = video.read()
        out_env_cat = env_cat(envs)
        cv2.imwrite(env_cat_path,out_env_cat)

if __name__ == '__main__':

    main()
