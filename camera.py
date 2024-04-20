# Modified by Fu Shuai for deploy
from cmath import inf
from dis import dis
from enum import Flag
from glob import glob
from http.client import NotConnected
from pickle import TRUE
from re import T
from sqlite3 import connect
from tkinter.tix import Tree
from typing_extensions import Self
import cv2
import sys
import time
import math
import copy
import json
import torch
import base64
import numpy as np
from socket import *
# import my_utils.my_detect as my_detect
from ultralytics import YOLO
from my_utils.stereo_config_200 import stereoCamera
import threading

# global settings
INTERVAL = 0.1  # 算法执行频率[s]
USE_SOCKET = False  # 是否需要对接服务器
SERVER_ADDR = ("127.0.0.1", 8899)  # 服务器ip及端口
LABELS = {0: "rivet", 1: "Aluminum chips",
          2: "Big Aluminum chips", 3: "screw", 4: "nuts"}  # classID : class_name
startDetect = True
serverDisconnect = False
not_connected = 1
oo = 1


class DetectResult:
    # type_name: detected object name
    # confidence: detecting confidence
    # box: detecting box [x_left_top, y_l_t, x_right_bottom, y_r_b]
    def __init__(self, type_name, confidence, box):
        self.type = type_name
        self.confidence = confidence
        self.box = box

    # rewrite print

    def __str__(self):
        return "box:{} confidence:[{:.2f}] type:[{}]".format(self.box, self.confidence, self.type)


class Camera:
    def __init__(self):
        print("Loading Net Model...")
        # my_detect.LoadModel()
        self.my_detect = YOLO("best.pt")
        self.config = stereoCamera()
        # self.left_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 打开摄像头，编号需根据PC/laptop修改
        # self.right_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.left_camera = cv2.VideoCapture("doubleCameraImages\\left_1.jpg")  # 打开摄像头，编号需根据PC/laptop修改
        self.right_camera = cv2.VideoCapture("doubleCameraImages\\right_1.jpg")

        # if open camera successfully,  print "open camera successfully!"
        if self.left_camera.isOpened():
            print("left camera openned successfully!")

        if self.right_camera.isOpened():
            print("right cameral openned successfully")
        self.W = 640
        self.H = 480

        # DetectResult list, for connect detecting and locating
        self.detect_results_left = []
        self.detect_results_right = []
        self.locate_results = []  # [[x,y,z], [x,y,z], ...] len is same as detect_results_left
        # [[x, y, z, x_pix, y_pix], ...] detected corners' location and pix coord in left frame
        self.corner_results = []
        print("Model and Camera Inited!")
        if USE_SOCKET:
            self.Connect()  # 连对接的服务器
            threading._start_new_thread(self.recieve, ())
        # T1.start()
        self.Loop()

    def __del__(self):
        # self.oneUSBCamera.release()
        self.left_camera.release()
        self.right_camera.release()
        cv2.destroyAllWindows()
        if USE_SOCKET:
            self.tcpClientSocket.close()
        print("Finish! Camera Released")

    def Detect(self, frame):
        # xyxyccs = my_detect.detect(frame)  # x1 y1 x2 y2 confidence class
        result = self.my_detect.predict(source=frame, save=False)
        xyxyccs = []

        for i in range(len(result)):
            boxes = result[i].boxes
            print(boxes)
            if len(boxes.conf) == 0:
                continue
            detect_results = []
            xyxycc = boxes.xyxy.tolist()[0]
            xyxycc.append(boxes.conf.tolist()[0])
            xyxycc.append(boxes.cls.tolist()[0])
            xyxyccs.append(xyxycc
                           )
        if len(xyxyccs) == 0:
            return []
        for xyxycc in xyxyccs:
            if xyxycc[4] < 0.5:  # confidence
                continue  # confidence
            x1 = int(xyxycc[0])
            y1 = int(xyxycc[1])
            x2 = int(xyxycc[2])
            y2 = int(xyxycc[3])
            confidence = float(xyxycc[4])
            classID = int(xyxycc[5])
            class_name = LABELS[int(xyxycc[5])]
            detect_results.append(DetectResult(
                class_name, confidence, [x1, y1, x2, y2]))
            # 画框写字
            # draw box and text
            if classID == 0 or classID == 3:
                color = [0, 0, 255]  # 铆钉 rivet R
            elif classID == 1:
                color = [0, 255, 0]  # 铝屑 Aluminum chips G
            elif classID == 2:
                color = [255, 0, 0]  # 大铝屑 Big Aluminum chips B
            elif classID == 3:
                color = [255, 0, 255]  # 螺栓 screw 紫色
            elif classID == 4:
                color = [255, 255, 0]  # 螺母 nuts 橘色
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 画框
            text = "{}: {:.4f}".format(class_name, confidence)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # 写字
        return detect_results

    # 使用左右两图对应的定位框，来计算视差以及定位信息，此算法对目标检测结果准确度要求较高
    def Locate(self):
        # 识别框匹配
        # 设置左右两个识别框为对应框的阈值[pix]，注意两图间视差(主要是delta_x)会随着距离变近而变大
        # set the threshold of the left and right detection boxes for the corresponding boxes [pix]
        BOX_MATCH_TORLERANCE_X = 100
        BOX_MATCH_TORLERANCE_Y = 40  # 两个参数需要根据使用情况调节，平衡一下召回率和准确率
        # if matched, the index of the left result and the index of the right result
        matched_index = {}  # 左右匹配上的识别框，key为左结果下标，value为右结果下标

        for index_l, result_l in enumerate(self.detect_results_left):
            for index_r, result_r in enumerate(self.detect_results_right):
                print('result_l:', result_l)
                print(result_l, result_r)
                if result_l.type != result_r.type:  # if the type is not the same, skip
                    continue  # 遍历时类型不一致直接跳过
                boxl = result_l.box
                boxr = result_r.box
                if abs(boxl[0] - boxr[0]) < BOX_MATCH_TORLERANCE_X and \
                        abs(boxl[1] - boxr[1]) < BOX_MATCH_TORLERANCE_Y and \
                        abs(boxl[2] - boxr[2]) < BOX_MATCH_TORLERANCE_X and \
                        abs(boxl[3] - boxr[3]) < BOX_MATCH_TORLERANCE_Y:
                    matched_index[index_l] = index_r
                    break
        # 对匹配上了的识别框进行双目视差定位
        # locate the matched detection boxes
        locate_results = []
        for index_l in range(len(self.detect_results_left)):  # 对每一个识别结果
            if index_l not in matched_index:  # if the left result is not matched, failed
                locate_results.append([10000, 10000, 10000])  # 定位失败
            else:
                boxl = self.detect_results_left[index_l].box
                boxr = self.detect_results_right[matched_index[index_l]].box
                # 根据双目测距原理计算X方向两边缘点坐标和中心坐标，改自之前matlab的实现
                # based on the principle of binocular ranging, calculate the coordinates of the X direction edges and the center coordinates
                pix_x_l = (boxl[0] + boxl[2]) / 2  # 像素坐标系下左图x中心
                # as the operation system team asked, change the y to the bottom of the box
                pix_y_l = boxl[3]  # )/2   # 应上交要求更改为像素坐标系下左图框的下沿y,
                pix_x_r = (boxr[0] + boxr[2]) / 2  # 像素坐标系下右图x中心
                pix_y_r = (boxr[1] + boxr[3]) / 2  # 像素坐标系下右图y中心
                X, Y, Z = self.StereoLocating(pix_x_l, pix_y_l, pix_x_r)
                locate_results.append([X, Y, Z])
        return locate_results

    # 对匹配的两点，从像素坐标系转换到图像坐标系，得到视差后计算相对于左相机光心的XYZ空间坐标
    # Convert from pixel coordinate system to image coordinate system for matched two points, calculate the XYZ space coordinates relative to the left camera center after obtaining the parallax
    def StereoLocating(self, pix_x_l, pix_y_l, pix_x_r):
        pix_x_l = pix_x_l * 225.45855 * 2 / 640
        pix_x_r = pix_x_r * 225.45855 * 2 / 640
        pix_y_l = pix_y_l * 230.8 * 2 / 480

        if pix_x_l - pix_x_r < 0:  # 像素视差[pix]
            print("[Error] 视差小于零！请检查左右镜头定义是否准确！")
        y_l = (pix_y_l - self.config.cy_l) / self.config.fy_l * \
              self.config.f  # the y coordinate of the left image coordinate system
        x_l = (pix_x_l - self.config.cx_l) / \
              self.config.fx_l * self.config.f  # the x coordinate of the left image coordinate system
        x_r = (pix_x_r - self.config.cx_r) / \
              self.config.fx_r * self.config.f  # the x coordinate of the right image coordinate system
        D = (x_l - x_r)  # the parallax

        Z = self.config.f * self.config.b / D
        if Z < 0:
            Z = - Z
        X = x_l * Z / self.config.f
        Y = y_l * Z / self.config.f

        return X, Y, Z

    # corner detection
    def CornerDetect(self, left_frame_org, right_frame_org, left_frame, right_frame):
        corner_results = []
        # *********** 左右关键点提取(得到角点的超集) ***********
        left_frame_gray = cv2.cvtColor(left_frame_org, cv2.COLOR_BGR2GRAY)
        right_frame_gray = cv2.cvtColor(right_frame_org, cv2.COLOR_BGR2GRAY)
        feature_points_l = cv2.goodFeaturesToTrack(
            left_frame_gray, maxCorners=40, qualityLevel=0.05, minDistance=30)
        feature_points_r = cv2.goodFeaturesToTrack(
            right_frame_gray, maxCorners=40, qualityLevel=0.05, minDistance=30)
        # *********** 左右直线检测(用于从角点超集中筛选得到角点) ***********
        edges_l = cv2.Canny(left_frame_gray, 10, 100)
        # cv2.imshow("edges", edges_l)  # for canny thresh adjust
        lines_l = cv2.HoughLinesP(
            edges_l, 1, np.pi / 180, 5, minLineLength=200, maxLineGap=40)
        edges_r = cv2.Canny(right_frame_gray, 10, 100)
        lines_r = cv2.HoughLinesP(
            edges_r, 1, np.pi / 180, 5, minLineLength=200, maxLineGap=40)
        # 这四个有一个为空就不可能检测到角点，可以直接退出了
        if type(feature_points_l) == type(None) or type(feature_points_r) == type(None) or \
                type(lines_l) == type(None) or type(lines_r) == type(None):
            return corner_results
        # *********** 直线精简(根据斜率k和b筛去相似的直线) ***********
        lines_l_s = self.SimplifyLines(lines_l)
        lines_r_s = self.SimplifyLines(lines_r)
        # *********** 角点筛选(根据特征点和直线来筛选角点) ***********
        corners_l = self.SelectCorners(feature_points_l, lines_l_s, left_frame)
        corners_r = self.SelectCorners(
            feature_points_r, lines_r_s, right_frame)
        # *********** 角点匹配(与定位中识别框匹配原理类似) ***********
        CORNER_MATCH_TORLERANCE_X = 80  # 认为两角点是对应角点的距离阈值
        CORNER_MATCH_TORLERANCE_Y = 15
        matched_index = {}  # 左右匹配上的角点，key为左结果下标，value为右结果下标
        for index_l, corner_l in enumerate(corners_l):
            for index_r, corner_r in enumerate(corners_r):
                if abs(corner_l[0] - corner_r[0]) < CORNER_MATCH_TORLERANCE_X and \
                        abs(corner_l[1] - corner_r[1]) < CORNER_MATCH_TORLERANCE_Y and \
                        corner_l[1] < 400:
                    matched_index[index_l] = index_r
        # *********** 角点定位(与定位原理类似) ***********
        for index_l in range(len(corners_l)):  # 对左图每个角点
            if index_l not in matched_index:  # 若左图中的角点在右图没有匹配上
                continue
            else:
                # (849,377)
                pix_x_l = corners_l[index_l][0]
                pix_y_l = corners_l[index_l][1]
                pix_x_r = corners_r[matched_index[index_l]][0]
                pix_y_r = corners_r[matched_index[index_l]][1]
                # print('\n\n\n')
                # print(pix_x_l, pix_y_l, pix_x_r)
                X, Y, Z = self.StereoLocating(pix_x_l, pix_y_l, pix_x_r)
                # print(X, Y, Z)
                corner_results.append([X, Y, Z, int(pix_x_l), int(pix_y_l)])
        # *********** 画图 ***********
        for c in corners_l:
            cv2.circle(left_frame, (c[0], c[1]), 5,
                       (0, 0, 255), 4)  # 画通过角点筛选算法筛出的角点
        for c in corners_r:
            cv2.circle(right_frame, (c[0], c[1]), 5, (0, 0, 255), 4)
        for l in lines_l_s:
            cv2.line(left_frame, (l[0], l[1]),
                     (l[2], l[3]), (0, 255, 0), 2)  # 画精简后的直线检测结果
        for l in lines_r_s:
            cv2.line(right_frame, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2)
        return corner_results

    def SimplifyLines(self, lines):
        # if two lines are similar, only keep one
        lines_simplified = []  # [x1, y1, x2, y2, k, b]
        LINE_SIMPLIFY_DIS_THRESH = 100  # the distance threshold of two lines
        LINE_SIMPLIFY_K_THRESH = 0.7  # the slope threshold of two lines
        LINE_SIMPLIFY_K_THRESH1 = 1.3
        LINE_SIMPLIFY_B_THRESH = 50
        for line in lines:
            x1, y1, x2, y2 = line[0]
            k = (y1 - y2) / (x1 - x2 + 1e-5)
            b = y1 - k * x1
            if_add = True
            for line_s in lines_simplified:  # if the line is similar to the existing lines, do not add
                x1_, y1_, x2_, y2_, k_, b_ = line_s
                dis1 = math.sqrt(pow(x1 - x1_, 2) + pow(y1 - y1_, 2))  # the distance of two points
                dis2 = math.sqrt(pow(x2 - x2_, 2) + pow(y2 - y2_, 2))
                if (dis1 < LINE_SIMPLIFY_DIS_THRESH or dis2 < LINE_SIMPLIFY_DIS_THRESH) and \
                        (abs(line_s[4] / k) > LINE_SIMPLIFY_K_THRESH and abs(line_s[4] / k) < LINE_SIMPLIFY_K_THRESH1):
                    if_add = False
                    break
            if if_add:
                lines_simplified.append([x1, y1, x2, y2, k, b])
        return lines_simplified

    def SelectCorners(self, feature_points, lines, frame_to_draw):
        # if a feature point is on a line, it is a corner
        P_TO_LINE_DIS_THRESH = 50  # the distance threshold of a point to a line
        corners = []
        for p in feature_points:
            x = np.int32(p[0][0])
            y = np.int32(p[0][1])
            count = 0
            for l in lines:
                dis = []

                # the distance of the point to the line
                dis.append(math.sqrt(pow(x - l[0], 2) + pow(y - l[1], 2)))
                dis.append(math.sqrt(pow(x - l[2], 2) + pow(y - l[3], 2)))
                if min(dis) < P_TO_LINE_DIS_THRESH:
                    count += 1
            if count >= 2:
                corners.append([x, y])
            cv2.circle(frame_to_draw, (x, y), 5, (255, 0, 0), 2)
        return corners

    def log_result_and_draw(self, left_frame):
        print("\r\n共识别出[", len(self.detect_results_left), "]个多余物")
        if len(self.detect_results_left) != 0:
            print("detect result:")
        for i in range(len(self.detect_results_left)):
            print(self.detect_results_left[i])
            locate_text = "x:{:.2f}, y:{:.2f}, z:{:.2f}".format(self.locate_results[i][0],
                                                                self.locate_results[i][1], self.locate_results[i][2])
            print(locate_text)
            cv2.putText(left_frame, locate_text,
                        (self.detect_results_left[i].box[0], self.detect_results_left[i].box[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)
        if len(self.corner_results) != 0:
            print("corner result:")
        for result in self.corner_results:
            locate_text = "x:{:.2f}, y:{:.2f}, z:{:.2f}".format(
                result[0], result[1], result[2])
            # print(locate_text)
            cv2.putText(left_frame, locate_text, (result[3], result[4] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)

    # connect to socket server
    def Connect(self):
        global serverDisconnect
        global not_connected
        self.tcpClientSocket = socket(AF_INET, SOCK_STREAM)
        not_connected = 1

        while not_connected:
            try:
                self.tcpClientSocket.connect(SERVER_ADDR)
                not_connected = 0

            except:
                print("Connecting failed, keep trying......")
            time.sleep(1)
        serverDisconnect = False
        print("Connected!")

    # send All
    def SendPackage(self, frame):
        global oo
        json_dict = {"imageb64": "",
                     "detect_results": [], "corner_results": []}
        # image
        encode = cv2.imencode(".jpg", frame)[1]  # 'array' of (68353, 1)
        base64_data = base64.b64encode(encode)  # 'bytes' of len 91140
        base64_str = str(base64_data, 'utf-8')  # 'str' of len 91140
        json_dict["imageb64"] = base64_str
        m = 1500
        # detect
        min_Dis_Result = {
            "class": "123",
            "confidence": "123",
            "box": "123",
            "location": "123"
        }
        if len(self.detect_results_left) != 0:
            for i, result in enumerate(self.detect_results_left):
                listR = []
                for items in self.locate_results[i]:
                    if not np.isinf(items) and items < 1000:
                        listR.append(round(items, 6))
                    else:
                        listR.append(10000)
                if listR[2] < m:
                    min_Dis_Result = {
                        "class": result.type,
                        "confidence": str(result.confidence),
                        "box": str(result.box),
                        "location": str(listR)
                    }
                    m = listR[2]

            if not "10000" in min_Dis_Result["location"] and not min_Dis_Result["class"] == "123":
                # print("locate failed")
                json_dict["detect_results"].append(min_Dis_Result)
            else:
                print("111111")
        # corner
        if len(self.corner_results) != 0:
            for result in self.corner_results:
                listR = []
                for items in result[0:3]:
                    if not np.isinf(items) and items < 1000:
                        listR.append(round(items, 6))
                    else:
                        listR.append(10000)
                result_dict = {
                    "pix": str(result[3:5]),
                    "location": str(listR)
                }
                if not "10000" in result_dict["location"]:
                    json_dict["corner_results"].append(result_dict)
        # dump and send
        sendData = json.dumps(json_dict)
        self.tcpClientSocket.send(sendData.encode('utf-8'))

    # main loop
    def Loop(self):
        global startDetect
        last_time = time.time() - INTERVAL
        a = True
        while True:
            if time.time() - last_time >= INTERVAL:
                last_time = time.time()
                # foo, left_frame = self.left_camera.read() #双usb用
                # foo, right_frame = self.right_camera.read()
                dir_path = 'my_utils/imgs/测试集图片/'
                left_frame = cv2.imread(dir_path + '586.jpg')
                right_frame = cv2.imread(dir_path + '587.jpg')
                if (startDetect):
                    left_frame, right_frame = self.config.Rectify(
                        left_frame, right_frame)  # rectify
                    # save the original frame
                    left_frame_org = copy.copy(left_frame)
                    right_frame_org = copy.copy(right_frame)
                    self.detect_results_left = self.Detect(left_frame)
                    self.detect_results_right = self.Detect(right_frame)
                    self.locate_results = self.Locate()
                    self.corner_results = self.CornerDetect(left_frame_org, right_frame_org,
                                                            left_frame, right_frame)
                    self.log_result_and_draw(left_frame)
                cv2.imshow("left", left_frame)
                cv2.imshow("right", right_frame)
                if USE_SOCKET and not not_connected:  # send package
                    # print("21312312")
                    if not_connected:
                        print("Reconnecting")
                    else:
                        self.SendPackage(left_frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    return

    def recieve(self):
        # global SERVER_ADDR
        global startDetect
        global serverDisconnect
        global not_connected
        while 1:
            C = self.tcpClientSocket.recv(1024)
            s = str(C, encoding="utf8")
            if "start" in s:

                startDetect = True
            elif "stop" in s:
                startDetect = False
            elif "discon" in s:
                serverDisconnect = 0
                not_connected = 1
            elif "connect" in s and serverDisconnect == 0:
                print("重新连接成功")
                serverDisconnect = 1
                not_connected = 0
            elif "quit" in s:
                quit()


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg == "-socket" or arg == "-s":
            USE_SOCKET = True
        else:
            print("[Error] Invalid argument input!")
            sys.exit()
    camera = Camera()
