import cv2
import math
import numpy as np

# 双目相机参数（通过Matlab标定工具箱获得）
class stereoCamera(object):
    def __init__(self):
        self.W = 640
        self.H = 480
        self.CMOS_SIZE = 1/13  # 相机底大小(感光元件尺寸)
        # 左相机内参
        self.cam_matrix_left = np.array([[489.0755, -0.4154, 312.3728],
                                         [0, 487.9886, 235.0986],
                                         [0, 0, 1]])
        # 右相机内参
        self.cam_matrix_right = np.array([[481.3449, 0.1387, 318.5965],
                                          [0, 480.4469, 231.4898],
                                          [0, 0, 1]])
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.0878, -0.1032, -0.0009, -0.0037, 0.0706]])
        self.distortion_r = np.array([[0.0776, -0.1479, 0.0036, -0.0012, 0.1447]])
        # 旋转矩阵
        self.R = np.array([[0.9861, -0.0864, -0.1419],
                           [0.0939, 0.9944, 0.0476],
                           [0.1370, -0.0603, 0.9887]])
        # 平移矩阵
        self.T = np.array([[-5.7920], [0.2223], [-1.1692]])
        # for location
        self.fx_l = float(self.cam_matrix_left[0][0])
        self.fx_r = float(self.cam_matrix_right[0][0])
        self.fy_l = float(self.cam_matrix_left[1][1])
        self.fy_r = float(self.cam_matrix_right[1][1])
        self.cx_l = float(self.cam_matrix_left[0][2])
        self.cx_r = float(self.cam_matrix_right[0][2])
        self.cy_l = float(self.cam_matrix_left[1][2])
        self.cy_r = float(self.cam_matrix_right[1][2])
        self.f = (self.fx_l + self.fy_l + self.fx_r + self.fy_r) / 4 /     \
                math.sqrt(640*640+480*480) * self.CMOS_SIZE*25.4  # 根据相机底尺寸，计算焦距
        self.b = float(self.T[0])

    # 图像畸变校正(常规单目畸变校正方法,不如Rectify)
    def Undistort(self, left_frame, right_frame):
        mapx_l, mapy_l = cv2.initUndistortRectifyMap(self.cam_matrix_left, \
                self.distortion_l, self.R, self.cam_matrix_left, \
                (self.W, self.H), 5)
        mapx_r, mapy_r = cv2.initUndistortRectifyMap(self.cam_matrix_right, \
                self.distortion_r, self.R, self.cam_matrix_right, \
                (self.W, self.H), 5)
        left_undistorted = cv2.remap(left_frame, mapx_l, mapy_l, cv2.INTER_LINEAR)
        right_undistorted = cv2.remap(right_frame, mapx_r, mapy_r, cv2.INTER_LINEAR)
        return left_undistorted, right_undistorted

    # 双目畸变校正和立体矫正(好用!)
    def Rectify(self, left_frame, right_frame):
        # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        map1x, map1y, map2x, map2y, Q = self.getRectifyTransform()
        left_rectified, right_rectified = self.rectifyImage(left_frame, right_frame, map1x, map1y, map2x, map2y)
        return left_rectified, right_rectified

    def getRectifyTransform(self):
        # 计算校正变换
        # R1为旋转矩阵，相机坐标系的校正；P1为新投影矩阵，即新相机坐标系点投影于图像坐标系；
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cam_matrix_left, \
                self.distortion_l, self.cam_matrix_right, self.distortion_r, \
                (self.W, self.H), self.R, self.T, alpha=0)
        map1x, map1y = cv2.initUndistortRectifyMap(self.cam_matrix_left, \
                self.distortion_l, R1, P1, (self.W, self.H), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.cam_matrix_right, \
                self.distortion_r, R2, P2, (self.W, self.H), cv2.CV_32FC1)
        # 根据畸变校正后的新内参矩阵更新定位算法需要的参数
        self.fx_l = P1[0][0]
        self.fx_l = P1[0][0]
        self.fx_r = P2[0][0]
        self.fy_l = P1[1][1]
        self.fy_r = P2[1][1]
        self.cx_l = P1[0][2]
        self.cx_r = P2[0][2]
        self.cy_l = P1[1][2]
        self.cy_r = P2[1][2]
        self.f = (self.fx_l + self.fy_l + self.fx_r + self.fy_r) / 4 /     \
                math.sqrt(640*640+480*480) * self.CMOS_SIZE*25.4
        return map1x, map1y, map2x, map2y, Q

    # 畸变校正和立体校正
    def rectifyImage(self, image1, image2, map1x, map1y, map2x, map2y):
        rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
        rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
        return rectifyed_img1, rectifyed_img2
