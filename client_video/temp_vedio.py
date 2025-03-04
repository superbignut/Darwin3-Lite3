"""
这个文件是用来 做 手势的初步测试的，因为以前一直是用 高翔的， 看看opencv 能不能做的简单一点
"""
import cv2 as cv
import numpy as np
import struct
import threading
import time
import os 
import sys
sys.path.append('.')

from opencv_zoo.models.person_detection_mediapipe.mp_persondet import MPPersonDet
# from opencv_zoo.models.object_detection_nanodet.nanodet import NanoDet
# from opencv_zoo.models.face_detection_yunet.yunet import YuNet
from opencv_zoo.models.palm_detection_mediapipe.mp_palmdet import MPPalmDet
from opencv_zoo.models.handpose_estimation_mediapipe.mp_handpose import MPHandPose

from utils.RoI import RoIHumanDetMP # , RoIObjDetNano, RoIFaceDetYuNet
from utils.HandGesture import HandGesture
# from Controller import Controller
from utils.ColorDetection import ColorDetection

Develop_Mode = True  # True means use computer camera. False means use dog camera

if __name__ == '__main__':

    # get raw video frame

    cap = cv.VideoCapture(0) # 这里测试了在狗上也能捕捉到



    # try to use CUDA
    if cv.cuda.getCudaEnabledDeviceCount() != 0:
        backend = cv.dnn.DNN_BACKEND_CUDA
        target = cv.dnn.DNN_TARGET_CUDA
    else:
        backend = cv.dnn.DNN_BACKEND_DEFAULT
        target = cv.dnn.DNN_TARGET_CPU
        print('CUDA is not set, will fall back to CPU.')

    # human detector, used to determine where a person is to reduce the area of interest
    human_detector_mp = MPPersonDet(
        modelPath='utils/person_detection_mediapipe_2023mar.onnx',
        nmsThreshold=0.3,
        scoreThreshold=0.3,  # lower to prevent missing human body
        topK=1,  # just only one person
        backendId=backend,
        targetId=target)
    # nano detector
    """     human_detector_nano = NanoDet(
        modelPath='utils/object_detection_nanodet_2022nov.onnx',
        prob_threshold=0.5,
        iou_threshold=0.6,
        backend_id=backend,
        target_id=target) """
    # face detector
    """     face_detector = YuNet(modelPath='utils/face_detection_yunet_2023mar.onnx',  # 2022-> 2023
                          confThreshold=0.6,  # lower to make sure mask face can be detected
                          nmsThreshold=0.3,
                          topK=5000,  # only one face
                          backendId=backend,
                          targetId=target) """
    # palm detector
    palm_detector = MPPalmDet(
        modelPath='utils/palm_detection_mediapipe_2023feb.onnx',
        nmsThreshold=0.3,
        scoreThreshold=0.4,  # lower to  prevent missing palms
        topK=5,  # maximum 2 palms to make sure right hand can be detected # origin=500
        backendId=backend,
        targetId=target)
    # handpose detector
    handpose_detector = MPHandPose(
        modelPath='utils/handpose_estimation_mediapipe_2023feb.onnx',
        confThreshold=0.6,  # higher to prevent mis-estimation
        backendId=backend,
        targetId=target)

    human_RoI_mp = RoIHumanDetMP(human_detector_mp) # 这两个都是检测人体 先用一个
    # human_RoI_nano = RoIObjDetNano(human_detector_nano)


    # face_RoI_yunet = RoIFaceDetYuNet(face_detector)
    hand_gesture = HandGesture(palm_detector, handpose_detector) # 这个使用 mediapipe 检测手势
    # mask_detector = ColorDetection(np.array([86, 28, 141]), np.array([106, 128, 225]))

    # gesture will be recognized only if the gesture is the same 2 times in a row
    gesture_buffer = [None] * 3
    while True:
        ret, frame = cap.read()
        if ret is None or not ret:
            continue

        # detect RoI by human detection
        bbox = human_RoI_mp.detect(frame)
        image = frame
        gestures = None

        if bbox is not None:

            upper_body_RoI = human_RoI_mp.get_upper_RoI() # 这里如果要使用全屏检测手势的话， 需要改成[[0,0],[640, 480]] 

            gestures, area_list = hand_gesture.estimate(frame, upper_body_RoI)
            # print(gestures, "1111") # gestures 是字符串的列表

            cloth = human_RoI_mp.get_cloth_RoI()
            # print(cloth)
            
            if cloth is not None:
                        
                # 重塑cloth为(2, 2)并确保为整数
                cloth = cloth.reshape(-1).astype(int)
                x1, y1, x2, y2 = cloth
                # print(x1, y1, x2, y2)
                
                # 提取布料的RoI
                cloth_image = frame[y1:y2, x1:x2]
    
                # 将RoI转换为HSV颜色空间
                image_rgb = cv.cvtColor(cloth_image, cv.COLOR_BGR2RGB)
    
                lower_red = np.array([100, 0, 0])   # 红色的最低范围
                upper_red = np.array([255, 80, 80]) # 红色的最高范围

                # 创建掩码
                red_mask = cv.inRange(image_rgb, lower_red, upper_red) # 在范围内的是255 其余变成0

                # 计算红色区域的像素数量
                red_pixels = cv.countNonZero(red_mask) # 计算非零区域

                # 计算总像素数量
                total_pixels = image_rgb.shape[0] * image_rgb.shape[1] # 统计总像素数

                # 计算红色占比
                red_ratio = int(red_pixels * 5 *100 / total_pixels) # 多乘了5 作为放大系数 python2 是整数
                if red_ratio > 8:

                    data = "color " + str(1) + " " + str(0) # 红色 发1 
                else:
                    data = "color " + str(0) + " " + str(0) # 其余
                
                # client_socket.sendall(data.encode('utf-8'))
                print( "红色占比, " , red_ratio)
                time.sleep(0.4)
                # cv.rectangle(image, (int(cloth[0]), int(cloth[1])), (int(cloth[2]), int(cloth[3])), (0, 255, 0), 1)

        cv.imshow("Demo", image)
        k = cv.waitKey(1) # 画图必备
        # control robot dog
        if gestures is not None and gestures.shape[0] != 0: # gestures有两个维度 第一个应该是 图像 第二个是分类结果
            # only use the biggest area right hand
            idx = area_list.argmax()
            gesture_buffer.insert(0, gestures[idx]) # 
            gesture_buffer.pop() # 每插入一个就要 pop一个
            # only if the gesture is the same 3 times, the corresponding command will be executed
            if gesture_buffer[0] is not None and all(ges == gesture_buffer[0] for ges in gesture_buffer):
                print(gesture_buffer[0])
                # 这里可以给socket 了
# 给models 加了一个init