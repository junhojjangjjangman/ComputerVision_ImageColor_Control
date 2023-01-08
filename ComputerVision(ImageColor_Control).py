import cv2
import numpy as np
import sys
import Dir
import pandas as pd
import matplotlib.pyplot as plt


img = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png")
Covert_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Covert_img = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png", 0)

cv2.imshow("img", img)
cv2.imshow("gray", Covert_img)# 회색조로 하는 이유는 데이터 처리량을 줄이기 위함
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.figure(figsize=(8,3))
plt.subplot(131);plt.imshow(Covert_img, cmap = 'gray') # cmap = 'gray'을 해주지 않으면 default값이 되어 gray 사진을 넣어도 gray색이 되지 않음
plt.subplot(132);plt.imshow(img)
plt.subplot(133);plt.imshow(Covert_img)
plt.show()

# 기본 임계처리(https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html)
# 함수원형: cv2.threshold(src, thresh, maxval, type) → retval, dst retval == return value
# Parameters:
# src – input image로 single-channel 이미지.(grayscale 이미지)
# thresh – 임계값
# maxval – 임계값을 넘었을 때 적용할 value
# type – thresholding type
    # cv2.THRESH_BINARY
    # cv2.THRESH_BINARY_INV
    # cv2.THRESH_TRUNC
    # cv2.THRESH_TOZERO
    # cv2.THRESH_TOZERO_INV

# 적응 임계처리
# 함수원형: cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
# Parameters:
# src – grayscale image
# maxValue – 임계값
# adaptiveMethod – thresholding value를 결정하는 계산 방법
    # cv2.ADAPTIVE_THRESH_MEAN_C : 주변영역의 평균값으로 결정
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C :
# thresholdType – threshold type
# blockSize – thresholding을 적용할 영역 사이즈
# C – 평균이나 가중평균에서 차감할 값


ret,thrsholded = cv2.threshold(Covert_img, 29,255,cv2.THRESH_BINARY) # 29~255사이에 있는 것을 흰색으로 바꿈
cv2.imshow("Thresholded",thrsholded)

ret,thrsholded2 = cv2.threshold(Covert_img, 29,255,cv2.THRESH_BINARY_INV) # 29~255사이에 있는 것을 흰색으로 바꿈
cv2.imshow("Thresholded2",thrsholded2)

cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(Dir.dir+'snowman.png')
canny = cv2.Canny(img, 150, 200)
        # 대상이미지, minVal(하위임계값), maxVal(상위임계값)
# cv2.imshow('img', img)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

def empty(pos):
    pass
img = cv2.imread(Dir.dir+'snowman.png')
name = 'Trackbar'
cv2.namedWindow(name)
cv2.createTrackbar('threshold1', name, 0, 255, empty)
cv2.createTrackbar('threshold2', name, 0, 255, empty)
while True:
    threshold1 = cv2.getTrackbarPos('threshold1', name)
    threshold2 = cv2.getTrackbarPos('threshold2', name)
    canny = cv2.Canny(img, threshold1, threshold2)
        # 대상이미지, minVal(하위임계값), maxVal(상위임계값)
    cv2.imshow('img', img)
    cv2.imshow(name, canny)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()