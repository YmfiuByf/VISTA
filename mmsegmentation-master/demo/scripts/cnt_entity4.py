# Copyright (c) OpenMMLab. All rights reserved.
Class=['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']
import sys
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
# sys.path.insert(0,r"D:\pycharmProject\paper\mmsegmentation-master")
import numpy as np
import os
import pandas as pd
sys.path.insert(0,os.getcwd()[:-5])
import cv2
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import goto
from dominate.tags import label
from goto import with_goto

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from argparse import ArgumentParser
# from env_cat import *
from tqdm import tqdm,trange
import matplotlib.pyplot as plt

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from PIL import Image

import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import cv2
import numpy as np
import spacy
nlp = spacy.load("en_core_web_lg")
def get_sim(str1,str2):
    doc1,doc2 = nlp(str1),nlp(str2)
    return doc1.similarity(doc2)
def switch2frame(d):
    ifr = d.find_element('xpath', '//*[@id="iFrameResizer0"]')
    d.switch_to.frame(ifr)

# print('Get current working directory : ', os.getcwd(),os.path.join(os.getcwd(),'figure.png'))

# out_path = os.path.join(os.getcwd(),'out.png')
out_path = r"D:\pycharmProject\paper\task2\out.png"
tmp_img_path = r"D:\pycharmProject\paper\task2\tmp.png"
def main(img_path,img_path2):
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
        result = result[0]
        a, indices = np.unique(result, return_counts=True)
        # print(a,indices)
        entity = ''
        for i in a:
            entity = entity+Class[i]+', '
        # for i in a:
        #     print(Class[i])
        print(entity[:-2])
        return a, entity[:-2]

    frame = cv2.imread(img_path)
    frame = cv2.resize(frame,(384,384))
    cv2.imwrite(tmp_img_path, frame)
    args.img = img_path
    a1,e1 = img_seg(model,tmp_img_path)
    frame = cv2.imread(img_path2)
    frame = cv2.resize(frame,(384,384))
    cv2.imwrite(tmp_img_path,frame)
    args.img = img_path2
    a2,e2 = img_seg(model,tmp_img_path)
    pre = np.intersect1d(a1,a2)
    pre_e = ''
    for i in pre:
        pre_e = pre_e + Class[i] + ', '
    return len(a1),e1,len(a2),e2,len(pre),pre_e[:-2]
# @goto.with_goto
# def refresh(driver,driver2):
#     label.begin
#     try:
#         iframe = driver.find_element('xpath', '//*[@id="iFrameResizer0"]')
#         iframe2 = driver2.find_element('xpath', '//*[@id="iFrameResizer0"]')
#         return iframe,iframe2
#     except:
#         driver.refresh()
#         driver2.refresh()
#         goto.begin

if __name__ == '__main__':
    flag = True
    parser = ArgumentParser()
    parser.add_argument('--img', default='', help='Image file')

    parser.add_argument('--config', default=os.path.join(os.getcwd()[:-5],
                                                         "configs\segmenter\segmenter_vit-l_mask_8x1_640x640_160k_ade20k.py"),
                        help='Config file')
    parser.add_argument('--checkpoint', default=os.path.join(os.getcwd()[:-5],
                                                             "configs\segmenter\segmenter_vit-l_mask_8x1_640x640_160k_ade20k_20220614_024513-4783a347.pth"),
                        help='Checkpoint file')
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
    num = 1000
    if os.path.exists(r"D:\pycharmProject\paper\task2\result4.csv"):
        df = pd.read_csv(r"D:\pycharmProject\paper\task2\result4.csv").iloc[:,1:]
    else:
        df = pd.DataFrame(columns=['caption_ori', 'caption_pd', 'similarity', 'num_entity',
                                   'entity_name1', 'num_entity2', 'entity_name2', 'num_pre', 'pre_entity'])
    driver = webdriver.Chrome()
    driver2 = webdriver.Chrome()
    driver.get("https://huggingface.co/spaces/stabilityai/stable-diffusion")
    # switch2frame(driver)
    driver2.get(
        'https://huggingface.co/spaces/flax-community/image-captioning#a-brown-and-white-horse-standing-next-to-a-fence')
    # switch2frame(driver2)
    # //*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[4]/div/div/p/a
    time.sleep(5)
    iframe = driver.find_element('xpath', '//*[@id="iFrameResizer0"]')
    iframe2 = driver2.find_element('xpath', '//*[@id="iFrameResizer0"]')
    driver.switch_to.frame(iframe)
    driver2.switch_to.frame(iframe2)
    def refresh():
        driver.refresh()
        driver2.refresh()
        time.sleep(20)
        flag = True
        while flag:
            try:
                iframe = driver.find_element('xpath', '//*[@id="iFrameResizer0"]')
                iframe2 = driver2.find_element('xpath', '//*[@id="iFrameResizer0"]')
                flag = False
            except NoSuchElementException:
                time.sleep(2)
                driver.refresh()
                driver2.refresh()
                time.sleep(20)
                iframe = driver.find_element('xpath', '//*[@id="iFrameResizer0"]')
                iframe2 = driver2.find_element('xpath', '//*[@id="iFrameResizer0"]')
        driver.switch_to.frame(iframe)
        driver2.switch_to.frame(iframe2)
    def ele_appear(d,expath,expath2=None):
        flag = True
        while flag:
            if expath2 is not None:
                d.find_element('xpath',expath2).click()
                time.sleep(10)
            try:
                d.find_element('xpath',expath)
                flag = False
            except NoSuchElementException or StaleElementReferenceException: pass
    def ele_disappear(d,expath,expath2=None):
        flag = True
        while flag:
            try: d.find_element('xpath',expath)
            except NoSuchElementException or StaleElementReferenceException: flag=False

    img1_xpath = '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[5]/div/div/div/img'
    # img1_xpath = '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[4]/div/div/p/a'
    img1_save_path = r"D:\pycharmProject\paper\task2\origin"
    xpath = '//*[@id="prompt-text-input"]/label/input'
    # //*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[5]/div/div/div/img
    bar_xpath = '//*[@id="gallery"]/div[1]/div[1]'
    change_xpath = '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[4]/div/button'
    for i in range(1000):
        # img1 = driver2.find_element('xpath',img1_xpath)
        print(i)
        try:
            img1 = WebDriverWait(driver2, timeout=90).until(lambda d: d.find_element(By.XPATH, img1_xpath))
        except TimeoutException:
            driver2.find_element('xpath',
                                           '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[4]/div/button/div/p').click()
            continue
        time.sleep(3)
        try:
            text = driver2.find_element('xpath',
                                        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[8]/div').text
        except NoSuchElementException:
            text = driver2.find_element('xpath',
                                         '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[7]/div/div/div').text
        for c in ['"','/','?','|',':','<','>']:
            if c in text: text = text.replace(c,' ')
        src1 = img1.get_attribute('src')
        if i<num:
            urllib.request.urlretrieve(src1, fr"D:\pycharmProject\paper\task2\origin\{text}.png")
        else:
            urllib.request.urlretrieve(src1, fr"D:\pycharmProject\paper\task2\no_space_origin.png")

        prompt = driver.find_element(By.XPATH, xpath)
        prompt.clear()
        prompt.send_keys(text)
        time.sleep(3)
        # driver.find_element(By.XPATH, '//*[@id="component-9"]').click()
        not_successfully_clicked = True
        while not_successfully_clicked:
            try:
                driver.find_element(By.XPATH, '//*[@id="component-9"]').click()
                time.sleep(1)
                driver.find_element('xpath', '//*[@id="gallery"]/div[1]/div[1]')
                not_successfully_clicked = False
            except NoSuchElementException: pass
        print('generating')
        ele_disappear(driver,bar_xpath)
        time.sleep(1)
        try:
            driver.find_element('xpath','//*[@id="gallery"]/div[1]/div/div[2]')
            time.sleep(15)
            print('problemetic keywords')
            refresh()
            continue
        except NoSuchElementException: pass
        try:
            img_path = WebDriverWait(driver, timeout=90).until(
                lambda d: d.find_element(By.XPATH, '//*[@id="gallery"]/div[2]/div/button[1]/img'))
            print('finish generation')
        except TimeoutException:
            print('problemetic keywords')
            refresh()
            continue

        src = img_path.get_attribute('src')
        if i<num:
            urllib.request.urlretrieve(src, fr"D:\pycharmProject\paper\task2\result2\{text}.png")
        else:
            urllib.request.urlretrieve(src, r"D:\pycharmProject\paper\task2\no_space_tmp.png")
        time.sleep(5)

        if i<num:  upload_path = fr"D:\pycharmProject\paper\task2\result2\{text}.png"
        else:  upload_path = r"D:\pycharmProject\paper\task2\no_space_tmp.png"

        try: driver2.find_element('xpath', '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[1]/div/div/ul/li/div')
        except NoSuchElementException: driver2.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[1]/div/section/input').send_keys(upload_path)
        not_uploaded = True
        print('uploading')
        while not_uploaded:
            driver2.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[2]/div/div/button/div/p').click()
            try:
                driver2.find_element('xpath', '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[1]/div/div/ul/li/div')
                # print('not uploaded')
            except NoSuchElementException:
                not_uploaded = False
                print('uploaded')
        text2_not_ready = True
        while text2_not_ready:
            try:
                driver2.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[8]/div/div/div').text
            except StaleElementReferenceException: pass
            except NoSuchElementException:
                text2 = driver2.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[7]/div/div/div').text
                text2_not_ready = False
        sim = get_sim(text, text2)
        print(text, text2, sim)

        # urllib.request.urlretrieve(src, fr"D:\pycharmProject\paper\task2\{tmp}.png")
        origin_path = fr"D:\pycharmProject\paper\task2\origin\{text}.png" if i<num else r"D:\pycharmProject\paper\task2\no_space_tmp.png"
        cnt1, entity1,cnt2, entity2,preserved,pre_entity = main(fr"D:\pycharmProject\paper\task2\origin\{text}.png",fr"D:\pycharmProject\paper\task2\result2\{text}.png")
        print('number of entity: ',cnt1, entity1,cnt2, entity2,preserved,pre_entity)
        # df = pd.DataFrame(columns=['caption_ori', 'caption_pd', 'similarity','num_entity',
        #                   'entity_name1','num_entity2','entity_name2','num_pre','pre_entity'])
        df.loc[len(df.index)] = [text,text2,sim,cnt1,entity1,cnt2,entity2,preserved,pre_entity]
        df.to_csv(r"D:\pycharmProject\paper\task2\result4.csv")
        # driver2.find_element('xpath',change_xpath).click()
        ele_appear(driver2,'//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[8]/div/div/div',change_xpath)
