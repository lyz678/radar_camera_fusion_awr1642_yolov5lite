import cv2
import time
import numpy as np
from v5lite import yolov5_lite


class yolov5_lite_realtime(yolov5_lite):
    def detect(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0]
        return outs

if __name__ == '__main__':
    modelpath = './cfg/v5lite-e_end2end.onnx'
    classfile = './cfg/coco.names'
    confThreshold = 0.5
    nmsThreshold = 0.6

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        exit()
    # 设置分辨率，640x480 1920*1080
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    net = yolov5_lite_realtime(modelpath, classfile, confThreshold=confThreshold, nmsThreshold=nmsThreshold)

    while True:
        t1 = time.time()
        ret, srcimg = cap.read()
        if not ret:
            break

        # 检测
        img, newh, neww, top, left = net.letterBox(srcimg.copy())
        outs = net.detect(img)
        boxes, indices, classIds, confidences = net.postprocess(srcimg, outs, (newh, neww, top, left))
        for ind in indices:
            srcimg = net.drawPred(srcimg, classIds[ind], confidences[ind], boxes[ind][0], boxes[ind][1], boxes[ind][2], boxes[ind][3])

        cost_time = time.time() - t1
        fps = 1.0 / cost_time  # 计算 FPS
        fps_text = f'FPS: {fps:.2f}'

        # 显示推理时间
        infer_time = f'Inference Time: {int(cost_time * 1000)} ms'
        cv2.putText(srcimg, infer_time, (5, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), thickness=1)

        # 显示 FPS
        cv2.putText(srcimg, fps_text, (5, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), thickness=1)

        # 显示图像
        cv2.imshow('YOLOv5 Lite Detection', srcimg)

        # 按 'q' 退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()
