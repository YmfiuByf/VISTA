import cv2 as cv
import numpy as np

img = cv.imread(r"D:\pycharmProject\video semantic communication\ball\seg\balla_1.png")
print(img)
green = img[:,:,1]>130
green = green.reshape((-1,1))
a, indices = np.unique(green, return_counts=True)
print(indices)


