import time
import numpy as np
import cv2
from radar2image import project_radar_points_to_camera_image
from readData_AWR1642 import readAndParseData16xx, serialConfig, parseConfigFile
from camera_realtime import yolov5_lite_realtime

# ------------------------------------------------------------------
# Function to update the data and display in the plot using OpenCV with camera srcimg
def update_radar_camera(dataOk, detObj, srcimg, boxes, indices, classes, classIds, confidences, camera_intrinsic_matrix, distortion_coefficients, rotation_matrix, translation_vector):
    if dataOk and len(detObj["x"]) > 0:
        # 将雷达点云转换为NumPy数组
        radar_points = np.column_stack((-detObj["x"], detObj["z"], detObj["y"]))  # xyz -> xzy
        
        # 将雷达点云投影到相机图像上
        pixel_coords = project_radar_points_to_camera_image(radar_points, camera_intrinsic_matrix, distortion_coefficients, rotation_matrix, translation_vector)
        pixel_coords = pixel_coords.reshape(-1, 2)
        
        # 计算缩放比例
        scale_x = srcimg.shape[1] / 1920  # 宽度缩放比例
        scale_y = srcimg.shape[0] / 1080  # 高度缩放比例
        
        # 缩放坐标
        scaled_pixel_coords = (pixel_coords * np.array([scale_x, scale_y])).astype(int)
        
        # 绘制点
        for i in range(len(scaled_pixel_coords)):
            u, v = scaled_pixel_coords[i]
            # 防止越界
            if u < 0 or u >= srcimg.shape[1] or v < 0 or v >= srcimg.shape[0]:
                continue
            cv2.circle(srcimg, (u, v), 5, (0, 255, 0), -1)
        
        # 对每个box去匹配scaled_pixel_coords中的radar点
        for idx in indices:
            classId = classIds[idx]
            confidence = confidences[idx]
            box = boxes[idx]
            x1, y1, x2, y2 = box
            mask = (scaled_pixel_coords[:, 0] >= x1) & (scaled_pixel_coords[:, 0] <= x2) & (scaled_pixel_coords[:, 1] >= y1) & (scaled_pixel_coords[:, 1] <= y2)
            points_in_box = scaled_pixel_coords[mask]
            
            if len(points_in_box) > 0:
                # 计算平均doppler和range
                avg_doppler = np.mean(detObj["doppler"][mask])
                avg_range = np.mean(detObj["range"][mask])
                
                # 计算角度并转换为度
                angles = np.arctan2(detObj["y"][mask], -detObj["x"][mask])  # 使用detObj中的x，y计算角度
                avg_angle_rad = np.mean(angles)  # 计算平均角度（弧度）
                avg_angle_deg = np.degrees(avg_angle_rad)  # 将弧度转换为度
                
                # 在目标框上显示classId, confidence, doppler, range和angle信息
                class_info = f"Class: {classes[int(classId)]}"
                confidence_info = f"Confidence: {confidence:.2f}"
                doppler_info = f"Doppler: {avg_doppler:.2f} m/s"
                range_info = f"Range: {avg_range:.2f} m"
                angle_info = f"Angle: {avg_angle_deg:.2f} deg"
                cv2.putText(srcimg, class_info, (x1+10, y1+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(srcimg, confidence_info, (x1+10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(srcimg, doppler_info, (x1+10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(srcimg, range_info, (x1+10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(srcimg, angle_info, (x1+10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                
                # 绘制目标框
                cv2.rectangle(srcimg, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return dataOk
# -------------------------    MAIN   -----------------------------------------  

if __name__ == "__main__":
    # Change the configuration file name
    configFileName = './cfg/1642config.cfg'   #1642config profile
    CLIport = {}
    Dataport = {}
    byteBuffer = np.zeros(2**15,dtype = 'uint8')
    byteBufferLength = 0
    camera_intrinsic_matrix = np.array([
        [1542.60291, 0, 979.284145],  # fx, 0, cx
        [0, 1543.39789, 657.131836],  # 0, fy, cy
        [0, 0, 1]                     # 0, 0, 1
    ])  # 相机内参矩阵
    distortion_coefficients = np.array([-0.37060392, 0.04148584, -0.00094008, -0.00232051, 0.05975395])  # 畸变系数
    rotation_matrix = np.eye(3)  # 相机旋转矩阵
    translation_vector = np.array([0, 0, 0])  # 相机平移向量
    modelpath = './cfg/v5lite-e_end2end.onnx'
    classfile = './cfg/coco.names'
    confThreshold = 0.5
    nmsThreshold = 0.6


    # 配置串口
    CLIport, Dataport = serialConfig(configFileName)

    # 从配置文件中获取参数
    configParameters = parseConfigFile(configFileName)

    # 启动摄像头
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        exit()


    # 设置分辨率，640x480 1920*1080
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    net = yolov5_lite_realtime(modelpath, classfile, confThreshold=confThreshold, nmsThreshold=nmsThreshold)

    while True:
        try:
            t1 = time.time()
            ret, srcimg = cap.read()
            if not ret:
                break

            # 检测
            img, newh, neww, top, left = net.letterBox(srcimg.copy())
            outs = net.detect(img)
            boxes, indices, classIds, confidences = net.postprocess(srcimg, outs, (newh, neww, top, left))        
            
            # 显示radar数据
            detObj = {}
            dataOk = 0
            
            # Read and parse the received data
            dataOk, frameNumber, detObj = readAndParseData16xx(Dataport, configParameters, byteBuffer, byteBufferLength)
            update_radar_camera(dataOk, detObj, srcimg, boxes, indices, net.classes, classIds, confidences, camera_intrinsic_matrix, distortion_coefficients, rotation_matrix, translation_vector)

            cost_time = time.time() - t1
            fps = 1.0 / cost_time  # 计算 FPS
            fps_text = f'FPS: {fps:.2f}'

            # 显示推理时间
            infer_time = f'Inference Time: {int(cost_time * 1000)} ms'
            cv2.putText(srcimg, infer_time, (5, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), thickness=1)

            # 显示 FPS
            cv2.putText(srcimg, fps_text, (5, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), thickness=1)

            #显示图像
            cv2.imshow('2D scatter plot with camera srcimg', srcimg)

            # time.sleep(0.03)  # 采样频率为 30Hz
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # 按下 Ctrl + C 停止程序并关闭所有资源
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            break

    # 释放所有资源
    cap.release()
    cv2.destroyAllWindows()


