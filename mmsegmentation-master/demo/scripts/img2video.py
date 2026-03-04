import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt

path = r"D:\pycharmProject\paper\DAVIS\JPEGImages\480p\monkeys"
out_path = path+ '\\video.avi'
# r"D:\pycharmProject\paper\DAVIS\JPEGImages\480p\monkeys"+ '\\video.avi'
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
flag = True
for file in os.listdir(path):
    if file[-3:] not in ['png','jpg']:
        continue
    file_path = os.path.join(path, file)
    img = np.array(Image.open(file_path))
    img = cv2.imread(file_path)
    if flag:
        #img_= np.array(img)
        size = ( img.shape[1], img.shape[0] )
        out = cv2.VideoWriter(out_path, fourcc, fps, size)
    flag = False
    out.write(img)



