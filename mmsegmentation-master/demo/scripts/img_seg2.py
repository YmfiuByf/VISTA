# Copyright (c) OpenMMLab. All rights reserved.
Class=['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']
import sys
# sys.path.insert(0,r"D:\pycharmProject\paper\mmsegmentation-master")
import numpy as np
import os
# print('Get current working directory : ', os.getcwd()[:-5])
sys.path.insert(0,os.getcwd()[:-5])
import cv2
import pandas as pd
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
img_path = os.path.join(os.getcwd(),'demo.png')    ### 这里是图片名称
# out_env = os.path.join(os.getcwd(),'out_env.png')
# out_beh = os.path.join(os.getcwd(),'out_beh.png')
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

    def get_contour(result, label):
        flag = False
        seg = result[:,:]
        pos = []
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                p = seg[i, j]
                if p == label:
                    if not flag:
                        flag = True  # left edge
                        pos.append([i,j])
                    elif flag:
                        seg[i, j] = -1
                elif p != label:
                    if not flag:
                        seg[i, j] = -1
                    elif flag:  # right edge
                        flag = False
                        pos.append([i,j])
        return seg,pos

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
        cv2.imwrite(r"D:\pycharmProject\paper\mmsegmentation-master\demo\segmentation.png",result)
        img = np.array(cv2.imread(tmp_img_path))
        # result = result.reshape((-1,1))
        a, indices = np.unique(result, return_counts=True)
        contour = np.zeros(result.shape,dtype=type(result[0][0]))
        obj_pos = {}
        df = pd.DataFrame()
        n_cord = 0
        for id in a:
            copy_res = result.copy()
            ct,pos = get_contour(copy_res, id)
            contour[ct!=-1] = 255
            ct[ct != -1] = 255
            ct[ct == -1] = 0
            obj_pos[Class[id]]=pos
            n_cord += len(pos)
            # pos = np.array(pos)
            # pos_x = list(pos[0])
            # pos_y = list(pos[1])
            # df = df.append(df2, ignore_index=True)
            # df.loc[len(df)] = [f'{Class[id]}_x']+pos_x
            #
            # print(len(pos))
            # plt.imshow(contour)
            # plt.show()
            cv2.imwrite(os.path.join(os.getcwd(),f'Contour_{Class[id]}.png'), ct)
        cv2.imwrite(os.path.join(os.getcwd(), f'Contour.png'), contour)
        print(f'number of coordinates={n_cord},bits={8*2*n_cord}')
        return

    frame = cv2.imread(img_path)
    frame = cv2.resize(frame,(256,256))
    cv2.imwrite(tmp_img_path, frame)
    img_seg(model,tmp_img_path)
    # plt.imshow(beh)
    # plt.show()

if __name__ == '__main__':

    main()
