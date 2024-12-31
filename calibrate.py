import numpy as np
import cv2 as cv
import glob

# 定义终止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备棋盘格角点的对象点
W_num = 7  # 横向角点数
H_num = 7  # 纵向角点数
objp = np.zeros((W_num * H_num, 3), np.float32)
objp[:, :2] = np.mgrid[0:W_num, 0:H_num].T.reshape(-1, 2)

# 用于存储所有图像的对象点和图像点的数组
objpoints = []  # 真实世界中的三维点
imgpoints = []  # 图像平面中的二维点

# 获取imgs目录下所有的图像文件路径
image_files = glob.glob('imgs/*.jpeg')  # 可以根据需要修改路径和扩展名

for fname in image_files:
    # 读取图像
    img = cv.imread(fname, cv.IMREAD_COLOR)
    if img is None:
        print(f"无法加载图像: {fname}")
        continue

    another = img.copy()

    # 将图像转换为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv.findChessboardCorners(gray, (W_num, H_num), None)

    # 如果找到，添加对象点和图像点（并对它们进行细化）
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 绘制并显示角点
        cv.drawChessboardCorners(img, (W_num, H_num), corners2, ret)
        cv.imshow(f"Chessboard {fname}", img)
        cv.waitKey(100)  # 每张图像显示100ms

# 进行相机标定
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印畸变参数和相机内参
print("Distortion Coefficients:\n", dist)
print("Camera Matrix:\n", mtx)

# 获取最优新相机矩阵
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

# 畸变校正
for fname in image_files:
    img = cv.imread(fname, cv.IMREAD_COLOR)
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # 显示原图和校正后的图像
    cv.imshow(f'Original Image: {fname}', img)
    cv.imshow(f'Undistorted Image: {fname}', dst)
    cv.waitKey(0)

cv.destroyAllWindows()
