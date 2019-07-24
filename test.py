import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로딩
orig_img = cv2.imread('./hangle3.png')

# 이미지 그레이 스케일 변환
grayImg = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

# 이미지 임계처리
ret, thresh = cv2.threshold(grayImg, 127, 255, 0)

# 이미지의 컨투어를 추출한다
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#image = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# 각 컨투어에 해당하는 영역을 사각형 좌표(x, y, w, h)들로 저장
rects = [cv2.boundingRect(each) for each in contours]
print('컨투어 영역 개수: ' + str(len(rects)))

# 추출된 사각형 영역 중 가로 & 세로 길이가 적당한 것들만 tmp2에 저장
tmp = [(x, y, w, h) for (x, y, w, h) in rects if ((w*h > 500)and(w*h < 5000))]
#print(tmp)
print('필터링된 컨투어 영역 갯수: ' + str(len(tmp)))

# 이미지에 rect를 그려준다
copy_image = orig_img.copy()
for rect in tmp:
    cv2.rectangle(copy_image, (rect[0], rect[1]),
                  (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)

cv2.imshow('global threshold', copy_image)
cv2.waitKey(0)

'''
adaptive thresholding 과 팽창 후 침식(닫기) 를 적용하여
글자의 전체적인 윤곽선을 좀 더 잘 추출하게 만든다
'''
print('-' * 50)

# 먼저 adaptive thresholding 을 적용
thresh2 = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
cv2.imshow('adaptive threshold', thresh2)
cv2.waitKey(0)

# 팽창 후 침식 => 닫기 : 전체적인 윤곽 파악
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 8))
opening = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
#closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

cv2.imshow('adaptive + opening', opening)
cv2.waitKey(0)

# 임계 처리 & 닫기 연산 적용 후 컨투어를 추출한다
contours2, hierarchy2 = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 각 컨투어에 해당하는 영역을 사각형 좌표(x, y, w, h)들로 저장
rects2 = [cv2.boundingRect(each) for each in contours2]
print('컨투어 영역 갯수: ' + str(len(rects2)))

# 추출된 사각형 영역 중 가로 & 세로 길이가 적당한 것들만 tmp2에 저장
tmp2 = [(x, y, w, h) for (x, y, w, h) in rects2 if ((w*h > 500)and(w*h < 50000))]
#print(tmp2)
print('필터링된 컨투어 영역 갯수: ' + str(len(tmp2)))

# 이미지에 rect를 그려준다
copy_image = orig_img.copy()
for rect in tmp2:
    cv2.rectangle(copy_image, (rect[0], rect[1]),
                  (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)

cv2.imshow('adaptive threshold + opening + blur', copy_image)
cv2.waitKey(0)