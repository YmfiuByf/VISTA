# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.insert(0,r"D:\pycharmProject\paper\mmsegmentation-master")
import numpy as np
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

img_path = r"D:\pycharmProject\video semantic communication\ball\balla_2.png"
out_path = "D:/pycharmProject/video semantic communication/ball/seg"+'/'  + img_path[-11:]
def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default=img_path,help='Image file')
    parser.add_argument('--config', default=r"D:\pycharmProject\paper\mmsegmentation-master\configs\segmenter\segmenter_vit-t_mask_8x1_512x512_160k_ade20k.py",help='Config file')
    parser.add_argument('--checkpoint', default =r"D:\pycharmProject\paper\segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth", help='Checkpoint file')
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

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    print(out_path)
    result = show_result_pyplot(
        model,
        args.img,
        result,
        # get_palette(args.palette),
        None,
        opacity=args.opacity,
        out_file=args.out_file)
    print(result)
    img = np.array(cv2.imread(args.img))
    # result = result.reshape((-1,1))
    a, indices = np.unique(result, return_counts=True)
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

    idx = find_k_max(indices,3)
    print(a,indices)
    idx1 = result!=a[idx]
    print(img.shape)
    img0 = img.copy()
    img0[idx1]=[0,0,0]
    imwrite_path = 'D:/pycharmProject/video semantic communication/ball/behavior/beh_' + img_path[-11:]
    cv2.imwrite(imwrite_path,img0)
    cv2.imshow('',img0)
    cv2.waitKey(0)


if __name__ == '__main__':

    main()
