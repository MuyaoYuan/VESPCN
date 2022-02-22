import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# 0号摄像头
while(cap.isOpened()):
    # 获取每一帧图像
    ret, frame = cap.read()
    # 如果获取成功显示图像
    if ret == True:
        cv.imshow('Cam', frame)
    # 每一帧间隔25ms
    '''
    cv2.waitKey()在有按键按下的时候返回按键的ASCII值, 否则返回-1
    & 0xFF的按位与操作只取cv2.waitKey(1)返回值最后八位, 因为有些系统cv2.waitKey()的返回值不止八位
    ord('q')表示q的ASCII值
    总体效果: 按下q键后break
    '''
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows