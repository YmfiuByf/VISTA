import scipy.misc
import cv2
from skimage.color import rgb2ycbcr,ycbcr2rgb
import numpy as np
from scipy.fftpack import dct,idct
from numpy import *
import numpy as np
from OFDM import *
import huffman
from collections import Counter
from test2 import *

def image2bits(img,SNRdb):
    w=8 #modify it if you want, maximal 8 due to default quantization table is 8*8
    w=max(2,min(8,w))
    h=w
    xLen = img.shape[1]//w
    yLen = img.shape[0]//h
    runBits=1 #modify it if you want
    bitBits=3  #modify it if you want
    rbBits=runBits+bitBits ##(run,bitSize of coefficient)
    useYCbCr=True #modify it if you want
    useHuffman=True #modify it if you want
    quantizationRatio=1 #modify it if you want, quantization table=default quantization table * quantizationRatio

    def myYcbcr2rgb(ycbcr):
        return (ycbcr2rgb(ycbcr).clip(0, 1) * 255).astype(np.uint8)

    originalImg = img.copy()
    ycbcr = rgb2ycbcr(img)
    rgb = myYcbcr2rgb(ycbcr)
    if (useYCbCr):
        img = ycbcr
    #     print(img,rgb)
    # cv2.imshow('',rgb)
    # cv2.waitKey()
    def toBlocks(img):
        blocks = np.zeros((yLen, xLen, h, w, 3), dtype=np.int16)
        for y in range(yLen):
            for x in range(xLen):
                blocks[y][x] = img[y * h:(y + 1) * h, x * w:(x + 1) * w]
        return np.array(blocks)

    blocks = toBlocks(img)
    print(blocks.shape)

    x1 = int(0.47 * xLen)  # x1,x2,y1,y2 is location of the face
    y1 = int(0.21 * yLen)
    if (useYCbCr):
        def ycbcrBlock2rgb(block):
            return (ycbcr2rgb(block) * (255 / ycbcr2rgb(block).max())).astype(np.uint8)

    def dctOrDedctAllBlocks(blocks, type="dct"):
        f = dct if type == "dct" else idct
        dedctBlocks = zeros((yLen, xLen, h, w, 3))
        for y in range(yLen):
            for x in range(xLen):
                d = zeros((h, w, 3))
                for i in range(3):
                    block = blocks[y][x][:, :, i]
                    d[:, :, i] = f(f(block.T, norm='ortho').T, norm='ortho')
                    if (type != "dct"):
                        d = d.round().astype(int16)
                dedctBlocks[y][x] = d
        return dedctBlocks

    dctBlocks = dctOrDedctAllBlocks(blocks, "dct")

    def blocks2img(blocks):
        W = xLen * w
        H = yLen * h
        img = zeros((H, W, 3))
        for y in range(yLen):
            for x in range(xLen):
                img[y * h:y * h + h, x * w:x * w + w] = blocks[y][x]
        return img

    newImg = blocks2img(dctBlocks)
    # print(newImg.shape)

    # quantization table
    QY = array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])
    QC = array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])
    QY = QY[:w, :h]
    QC = QC[:w, :h]
    qDctBlocks = copy(dctBlocks)
    Q3 = moveaxis(array([QY] + [QC] + [QC]), 0, 2) * quantizationRatio if useYCbCr else dstack(
        [QY * quantizationRatio] * 3)  # all r-g-b/Y-Cb-Cr 3 channels need to be quantized
    Q3 = Q3 * ((11 - w) / 3)
    # print(f'Q3={Q3}')
    qDctBlocks = (qDctBlocks / Q3).round().astype('int16')
    qqq = copy(qDctBlocks)
    # qDctBlocks = np.array([1,2,3,4])
    shape = qDctBlocks.shape
    tmp = qDctBlocks.reshape(-1,)+128
    # print(tmp.shape)
    # code,np_code,dict = my_huffman_encoding(tmp)
    tmp = int2bits(tmp)[0]
    # tmp = long_polar(tmp,SNRdb)
    tmp = OFDM(tmp,SNRdb=SNRdb)
    # tmp = my_huffman_decoding(tmp,dict)
    qDctBlocks = np.array(byte2int(tmp)) -128
    print(f'out={qDctBlocks}')
    qDctBlocks = qDctBlocks.reshape(shape)
    # tmp = OFDM(tmp,SNRdb) - 128
    # print(tmp)

    qDctImg = blocks2img(qDctBlocks).astype('int16')
    # print(qDctBlocks.shape)
    dedctBlocks = dctOrDedctAllBlocks(qDctBlocks * Q3, "idct")
    out_img = myYcbcr2rgb(blocks2img(dedctBlocks))
    # dedctBlocks2 = dctOrDedctAllBlocks(qqq * Q3, "idct")
    # out_img2 = myYcbcr2rgb(blocks2img(dedctBlocks2))
    # print(f'out_img2={out_img2}')
    # return out_img
    # print(out_img)
    # cv2.imshow('out',out_img)
    # cv2.waitKey()
    return out_img

def main(k,snr):
    video_path = r"D:\pycharmProject\paper\videos\final_street\street1_resized.avi"
    out_path = f"D:\\pycharmProject\\paper\\videos\\final_street\\stree_Polar_SNR={snr}_int{k}.jpg"
    video = cv2.VideoCapture(video_path)
    _, img = video.read()
    out_img = image2bits(img,SNRdb=snr)
    cv2.imwrite(out_path,out_img)
    img = img.reshape((-1,))
    out_img = out_img.reshape((-1,))
    mse = np.sum((img-out_img) ** 2) / len(img)
    psnr = 10*log((255*255/mse),10)
    print(f'k={k},snr={snr},psnr={psnr}')

if __name__=='__main__':
    # for k in [0]:
    #     for snr in [-6,-3,0,3,6,9,12]:
    #         main(k,snr)
    img_path = r"D:\pycharmProject\paper\Dynamic_JSCC-main\a large brown bear walking through a forest.jpg"
    img = cv2.imread(img_path)
    out_img = image2bits(img, SNRdb=0)
    # cv2.imwrite(out_path, out_img)
    img = img.reshape((-1,))
    out_img = out_img.reshape((-1,))
    mse = np.sum((img - out_img) ** 2) / len(img)
    psnr = 10 * log((255 * 255 / mse), 10)
    print(psnr)
    # print(f'k={k},snr={snr},psnr={psnr}')

