import cv2 as cv
import numpy as np

cap = cv.VideoCapture('../sense/1202/T04.avi')

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
videowriter = cv.VideoWriter('../sense/T04_OF_M.avi', fourcc, 30, (1280, 720))

while(1):
    ret, frame2 = cap.read()
    if not ret:
        break
    nxt = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)

    # 使用迭代Gunner Farneback 方法计算密集特征的光流
    # - prvs: previous grayscale frame
    # - next: next grayscale frame
    # - flow: None
    # - pyr_scale: 0.5经典金字塔，构建金字塔缩放scale
    # - level：3 No. of level of pyramid
    # - winsize：3 平均窗口大小，数值越大，算法对图像的鲁棒性越强
    # - iterations：15 迭代次数
    # - poly_n：5 像素邻域的参数多边形大小，用于在每个像素中找到多项式展开式；较大的值意味着图像将使用更平滑的曲面进行近似，从而产生更高的分辨率、鲁棒算法和更模糊的运动场；通常多边形n=5或7。
    # - poly_sigma：1.2 高斯标准差，用于平滑导数
    # - flags: 可以是以下操作标志的组合：OPTFLOW_USE_INITIAL_FLOW：使用输入流作为初始流近似值。OPTFLOW_FARNEBACK_GAUSSIAN: 使用GAUSSIAN过滤器而不是相同尺寸的盒过滤器；

    flow = cv.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow('frame2', bgr)
    videowriter.write(bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('../sense/opticalfb.png', frame2)
        cv.imwrite('../sense/opticalhsv.png', bgr)
    prvs = nxt

cap.release()
cv.destroyAllWindows()
videowriter.release()