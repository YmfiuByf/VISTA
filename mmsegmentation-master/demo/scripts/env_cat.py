import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt

path = r"D:\pycharmProject\video semantic communication\ball\env"
video_path = r"D:\pycharmProject\paper\videos\street2.avi"
env_path = video_path[:-4]+'_environment.jpg'

def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    envs = []
    while(success):
        envs.append(frame)
        success,frame = video.read()
    return envs

def get_imgs(path):
    imgs = []
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        img = Image.open(file_path)
        imgs.append(np.array(img))
    return imgs

def img_stitching(beh, env):
    ret = beh.copy()
    threshold = 20
    idx1 = beh[:,:,0]<env[:,:,0]
    idx2 = beh[:,:,1]<env[:,:,1]
    idx3 = beh[:,:,2]<env[:,:,2]
    idx = idx1 & idx2 & idx3
    ret[idx] = env [idx]
    return ret

def video2frames(video_path):
    video = cv2.VideoCapture(video_path)
    ret = []
    success,frame = video.read()
    while(success):
        ret.append(frame)
        success,frame = video.read()
    return ret

def env_cat(imgs):
    # print(type(imgs[0]))
    h,w = imgs[0].shape[0],imgs[0].shape[1]
    ret = np.zeros_like(imgs[0])
    for i in range(h):
        for j in range(w):
            pixel = [0,0,0]
            for img in imgs:
                if all(img[i,j]!=[0,0,0]):
                    pixel = img[i,j]
                    break
            ret[i,j] = pixel
    return ret

def env_cat2(imgs):
    def img_stitching(beh, env):
        ret = beh.copy()
        threshold = 20
        idx1 = beh[:, :, 0] < env[:, :, 0]
        idx2 = beh[:, :, 1] < env[:, :, 1]
        idx3 = beh[:, :, 2] < env[:, :, 2]
        idx = idx1 & idx2 & idx3
        ret[idx] = env[idx]
        return ret
    img = imgs[0]
    for i in range(1,len(imgs)):
        img = img_stitching(img,imgs[i])
    return img

# imgs = video2frames(video_path)
# env_cat = env_cat(imgs)
# image_pil=Image.fromarray(env_cat)
# save_path = video_path + 'env_cat.jpg'
# image_pil.save(save_path)
envs = read_video(video_path)
out = env_cat2(envs)
cv2.imwrite(env_path,out)
