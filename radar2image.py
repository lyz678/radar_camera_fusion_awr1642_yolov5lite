import numpy as np
import cv2

def project_radar_points_to_camera_image(radar_points, camera_intrinsic_matrix, distortion_coefficients, rotation_matrix, translation_vector):
    """
    将雷达点云投影到相机图像上的2D像素坐标，并进行畸变校正

    :param radar_points: 雷达点云，形状为 (N, 3) 的 NumPy 数组
    :param camera_intrinsic_matrix: 相机内参矩阵，3x3 的 NumPy 数组
    :param distortion_coefficients: 畸变系数，5元素的 NumPy 数组
    :param rotation_matrix: 相机旋转矩阵，3x3 的 NumPy 数组
    :param translation_vector: 相机平移向量，形状为 (3,) 的 NumPy 数组

    :return: 投影并畸变校正后的2D像素坐标，形状为 (N, 2) 的 NumPy 数组
    """
    # 将雷达点从雷达坐标系转换到相机坐标系
    camera_points = np.dot(radar_points, rotation_matrix.T) + translation_vector

    # 将相机坐标系的点投影到像素坐标
    pixel_coords = camera_points @ camera_intrinsic_matrix.T

    # 透视除法，得到2D像素坐标
    u = pixel_coords[:, 0] / pixel_coords[:, 2]
    v = pixel_coords[:, 1] / pixel_coords[:, 2]

    # 将齐次坐标转换为2D像素坐标
    pixel_coords_2d = np.column_stack((u, v))

    # 对2D像素坐标进行畸变校正
    undistorted_pixel_coords = cv2.undistortPoints(pixel_coords_2d, camera_intrinsic_matrix, distortion_coefficients, P=camera_intrinsic_matrix)

    # 返回畸变校正后的2D像素坐标
    return undistorted_pixel_coords



if __name__ == '__main__':
    # 示例数据
    radar_points = np.array([[0.1, 0, 0.5], [0.1, 0, 0.1], [0.1, 0, 0.5]])  # 示例雷达点云xyz
    camera_intrinsic_matrix = np.array([
        [1542.60291, 0, 979.284145],  # fx, 0, cx
        [0, 1543.39789, 657.131836],  # 0, fy, cy
        [0, 0, 1]                     # 0, 0, 1
    ])  # 示例内参矩阵

    distortion_coefficients = np.array([-0.37060392, 0.04148584, -0.00094008, -0.00232051, 0.05975395])  # 畸变系数
    rotation_matrix = np.eye(3)  # 假设为单位矩阵，表示没有旋转
    translation_vector = np.array([0, 0, 0])  # 假设无平移

    # 调用函数
    pixel_coords = project_radar_points_to_camera_image(radar_points, camera_intrinsic_matrix, distortion_coefficients, rotation_matrix, translation_vector)

    # 输出结果
    print("Projected and undistorted 2D pixel coordinates:")
    print(pixel_coords)