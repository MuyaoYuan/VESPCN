import numpy as np
import cv2 as cv

cap = cv.VideoCapture('kanna.mp4')

# 获取图像的宽和高
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# print(frame_width, frame_height)
fps = 60
'''
TODO: 后缀和编解码可以不对应吗?
VideoWriter_fourcc: 视频编解码器4字节代码
CV_FOURCC('P','I','M','1') = MPEG-1 codec
CV_FOURCC('M','J','P','G') = motion-jpeg codec
CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec
CV_FOURCC('U', '2', '6', '3') = H263 codec
CV_FOURCC('I', '2', '6', '3') = H263I codec
CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec
'''
out = cv.VideoWriter('outpy.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

while(cap.isOpened()):
    # 获取每一帧图像
    ret, frame = cap.read()
    # 如果获取成功显示图像
    if ret == True:
        # 将每一帧写入到输出文件中
        out.write(frame)
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()