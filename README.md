从awr1642中获取毫米波雷达点云数据，将其投影到图像平面，与yolov5lite目标检测框进行匹配，实现对目标的测距测速测角。算法易于部署到raspberrypi4b上，实现实时的目标检测。

![Image text](https://www.ti.com/content/dam/ticom/images/products/ic/sensing-products/evm-boards/awr1642boost-top.png)

# 环境依赖
- onnx
- pyserial
- onnxruntime
- numpy
- opencv-python
- PyQt5
- ...


# 相机标定得到内参矩阵
- 将摄像头采集的图像放入imgs文件夹
```bash
python calibrate.py
```
- 用得到的标定结果替换main.py中的camera_intrinsic_matrix，distortion_coefficients
  


# 连接awr1642实测
```bash
python main.py
```
点击这里观看视频](https://github.com/lyz678/Gesture-recognition-awr1642boost-raspberrypi4b/blob/main/models/%E5%BD%95%E5%B1%8F%202024-12-27%2021-16-12.mp4)

# 在raspberrypi4b上运行

- 将main.py及其依赖文件下载到raspberrypi4b上

```bash
python main.py
```
# 1642串口数据读取参考
- https://github.com/ibaiGorordo/AWR1642-Read-Data-Python-MMWAVE-SDK-2
  
# yolov5lite推理参考
- [https://github.com/vilari-mickopf/mmwave-gesture-recognition](https://github.com/ppogg/YOLOv5-Lite)






