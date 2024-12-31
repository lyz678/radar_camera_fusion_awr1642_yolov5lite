import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置棋盘格的尺寸 (内部角点数量)
W_num = 8  # 横向角点数
H_num = 8  # 纵向角点数

# 方块的大小
square_size = 200

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 绘制整个画布的白色背景
canvas_size = (W_num + 2) * square_size  # 包括外围一圈的总宽度
ax.add_patch(patches.Rectangle((0, 0), canvas_size, canvas_size, facecolor='white'))

# 计算棋盘格的起始坐标，使其居中
start_x = (canvas_size - W_num * square_size) / 2
start_y = (canvas_size - H_num * square_size) / 2

# 绘制棋盘格的黑白方块
for y in range(H_num):
    for x in range(W_num):
        # 计算方块的颜色
        color = 'black' if (x + y) % 2 == 0 else 'white'
        
        # 使用 patches.Rectangle 绘制矩形（左下角坐标，宽度，高度）
        ax.add_patch(patches.Rectangle((start_x + x * square_size, start_y + y * square_size), square_size, square_size, facecolor=color))

# 设置坐标轴属性
ax.set_xlim(0, canvas_size)
ax.set_ylim(0, canvas_size)
ax.set_aspect('equal')
ax.axis('off')  # 关闭坐标轴

# 保存为高清矢量图格式（SVG）
plt.savefig("chessboard_vector_with_white_bg.svg", format='svg', bbox_inches='tight', pad_inches=0)

# 保存为PNG格式
plt.savefig("chessboard_with_white_border.png", format='png', bbox_inches='tight', pad_inches=0, dpi=300)  # 可以指定dpi来控制图像质量

# 显示图像
plt.show()