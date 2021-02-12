import cv2 as cv
import numpy as np
import json
file_path = "../result/"
file = open('../picture_list.txt')
low_green = np.array([35, 70, 70])
high_green = np.array([70, 200, 200])
with open("result.json", "w+") as result_json:
    # 对于每一幅图像，
    for line in file.readlines():
        pic_path, lane_num = line.strip().split(" ")
        path = file_path + pic_path
        print(path, " ", lane_num)
        src = cv.imread(path)
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, low_green, high_green)
        num = int(lane_num)
        dic = {}
        lanes = []
        up_x = []
        j = 10
        while j < 1280:
            if mask[280, j] == 255:
                up_x.append(j)  # 记录上一层的纵坐标
                cv.circle(src, (j, 280), 2, (0, 255, 255), 1, 1)
                cv.imshow("mask", mask)
                cv.imshow("raw_image", src)
                cv.waitKey(1000)
                j += 15
            else:
                j += 1
        print(up_x)
        for i in range(0, num):  # 对于图片中的第i条直线
            lane = []
            for y in range(160, 280, 10):
                x = -2
                lane.append(x)
            for y in range(280, 720, 10):
                x = 10
                while x < 1280:
                    if  280 <= y < 400:
                        if up_x[i] - 10 < x < up_x[i] + 10 and mask[y, x] == 255:
                            lane.append(x)
                            up_x[i] = x
                            x += 15
                        else:
                            x += 1
                    elif 400 <= y < 500:
                        if up_x[i] - 20 < x < up_x[i] + 20 and mask[y, x] == 255:
                            lane.append(x)
                            up_x[i] = x
                            x += 20
                        else:
                            x += 1
                    elif 400 <= y < 720:
                        if up_x[i] - 20 < x < up_x[i] + 20 and mask[y, x] == 255:
                            lane.append(x)
                            up_x[i] = x
                            x += 20
                        else:
                            x += 1
            while len(lane) < 56:
                lane.append(-2)
            lanes.append(lane)
        dic["lanes"] = lanes
        dic["h_sample"] = [i for i in range(160, 720, 10)]
        dic["raw_file"] = pic_path
        dic["run_time"] = 10
        pic_json = json.dumps(dic).replace("\\", "") + "\n"
        result_json.write(pic_json)

    '''
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
    '''
