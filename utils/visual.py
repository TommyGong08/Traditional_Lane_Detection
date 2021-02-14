import cv2 as cv
import numpy as np
low_green = np.array([40, 40, 40])
high_green = np.array([65,200,200])
file_path = "../result/"
file = open('./picture_list.txt')
low_green = np.array([35, 70, 70])
high_green = np.array([70, 200, 200])
for line in file.readlines():
    pic_path, lane_num = line.strip().split(" ")
    path = file_path + pic_path
    print(path, " ", lane_num)
    src = cv.imread(path)
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, low_green, high_green)
    j = 10
    for i in range(280, 720, 10):
        j = 10
        while j < 1280:
            if mask[i, j] == 255:
                cv.circle(src, (j, i), 1, (0, 255, 255), 1, 1)
                j += 20
            else:
                j += 1
    cv.imshow("mask", mask)
    cv.imshow("raw_image", src)
    cv.waitKey(1000)
