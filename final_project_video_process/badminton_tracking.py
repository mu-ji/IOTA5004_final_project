import cv2
import numpy as np
import matplotlib.pyplot as plt

# 输入视频文件路径
input_video_path = "final_project_video_process/badminton_cut.mp4"

# 输出视频文件路径
output_video_path = "final_project_video_process/badminton_label.mp4"

lower_color = np.array([100],dtype=np.uint8)
upper_color = np.array([255],dtype=np.uint8)

cap = cv2.VideoCapture(input_video_path)
print(cap.isOpened())

# 获取输入视频的帧率、尺寸和编解码器
fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*"mp4v")  # 输出视频编解码器，这里使用MP4V编码器

# 创建VideoWriter对象用于写入输出视频
out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

badminton_trajectory = []
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        np.save("final_project_video_process/badminton_trajectory.npy", badminton_trajectory)
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary_image = cv2.threshold(blurred, 185, 255, cv2.THRESH_BINARY)
    #binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 将彩色图像转换到HSV颜色空间

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 定义白色的HSV阈值范围
    lower_white = np.array([10, 50, 190], dtype=np.uint8)
    upper_white = np.array([40, 100, 240], dtype=np.uint8)
    
    # 根据阈值提取白色部分
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # 创建一个全黑的图像，与原始图像大小相同
    binary_image = np.zeros_like(frame)

    # 将白色部分复制到黑色图像上
    binary_image[white_mask > 0] = 255#frame[white_mask > 0]

    kernel_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # 进行开运算
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    opened_image = np.array(opened_image, np.uint8)
    print(opened_image.shape)
    opened_image=opened_image[:,:,0]
    # 寻找轮廓
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算轮廓的几何中心
    if len(contours) > 0:
        contour = contours[0]  # 假设只有一块白色区域
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])  # X坐标
        cY = int(M["m01"] / M["m00"])  # Y坐标
        print("白色区域的几何中心坐标：({}, {})".format(cX, cY))
        badminton_trajectory.append([cX,cY])
    else:
        print("未找到白色区域")


    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

    out.write(frame)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #plt.imshow(frame)
    #plt.show()
    #cv2.imshow('gray',blurred)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break