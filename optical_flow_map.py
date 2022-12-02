import cv2
import numpy as np

# 绘制光流场图像
def drawOptFlowMap(flow,cflowmap,step,color):
    # 以 step 步长遍历图像
    for x in range(0,flow.shape[1],step):
        for y in range(0,flow.shape[0],step):
            # 绘制跟踪线
            cv2.line(cflowmap, (x,y), (int(x+flow[y][x][0]),int(y+flow[y][x][1])),color)
            cv2.circle(cflowmap,(x, y), 2, color, -1)
    return cflowmap
cap = cv2.VideoCapture('../sense/1126.avi')
# 得到视频的高度
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# 得到视频的宽度
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# 得到视频的帧数
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# 得到视频的帧速
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义写入视频的编码格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')  #  *'XVID' or *'MJPG'  写入视频类的编码格式
# 创建写入视频的类
output_video = cv2.VideoWriter("../sense/GF_method_output.avi", fourcc, int(fps), (int(width), int(height)))

# 读取第一帧图像
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if ret is False:
        break
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    # 使用稠密光流算法计算光流
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # 通过得到的光流绘制图像
    bgr = drawOptFlowMap(flow,frame2,16,(0,255,0))
    # cv2.imwrite("GF_output.jpg",bgr)
    output_video.write(bgr)
    # 更新
    prvs = next
cap.release()