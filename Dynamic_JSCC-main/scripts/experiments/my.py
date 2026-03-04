import cv2
import os
from matplotlib import pyplot as plt

path = r"D:\pycharmProject\paper\task2"
for file in os.listdir(path):
    if file[-3:] not in ['jpg','png']: continue
    file = os.path.join(path,file)
    img = cv2.imread(file)
    h, w, _ = img.shape
    print(h,w)
    plt.imshow(img)
    plt.show()
    if h==w: continue
    x = int(input())
    print(type(x))
    if h<w:
        img = img[:,x:x+h,:]
    elif h>w:
        img = img[x:x+w,:,:]
    plt.imshow(img)
    plt.show()
    # x = input()
    # img = cv2.resize(img,(256,256))
    cv2.imwrite(file,img)

