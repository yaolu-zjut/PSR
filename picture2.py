import torch
import matplotlib.pyplot as plt

# 读取每个 CKA 相似矩阵，并假设每个加载的对象是一个链表
cnn1d_cka = torch.load("/public/ly/zyt/xianyu/CKA_matrix_for_visualization_CNN1D_all_radio128.pth")
resnet110_cka = torch.load("/public/ly/zyt/xianyu/CKA_matrix_for_visualization_ResNet110_signal_all_radio128.pth")
resnet56_cka = torch.load("/public/ly/zyt/xianyu/CKA_matrix_for_visualization_ResNet56_signal_all_radio128.pth")
signet50_cka = torch.load("/public/ly/zyt/xianyu/CKA_matrix_for_visualization_SigNet50_all_radio128.pth")
vgg16_cka = torch.load("/public/ly/zyt/xianyu/CKA_matrix_for_visualization_vgg16_signal_all_radio128.pth")

# 假设每个 CKA 矩阵是一个链表
cka_matrices = [cnn1d_cka, resnet110_cka, resnet56_cka, signet50_cka, vgg16_cka] 

# 创建一个包含2行3列的图形
fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))

# 定义每个子图的标题
titles = ['CNN1D_RML2016.10a', 'ResNet110_RML2016.10a', 'ResNet56_RML2016.10a', 'SigNet50_RML2016.10a', 'VGG16_RML2016.10a']

# 绘制每个模型的 CKA 相似矩阵
for i, (ax, cka_matrix, title) in enumerate(zip(axs.flat, cka_matrices, titles)):
    # 假设 cka_matrix 是一个链表，且需要绘制每个链表中的矩阵
    for matrix in cka_matrix:
        im = ax.imshow(matrix, cmap='hot', interpolation='nearest')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Layer ID')
        ax.set_ylabel('Layer ID')
        fig.colorbar(im, ax=ax, shrink=0.8)  # 调整颜色条的大小
        break  # 只绘制链表中的第一个矩阵

# 删除多余的子图
for j in range(len(cka_matrices), len(axs.flat)):
    fig.delaxes(axs.flat[j])

# 调整布局以避免子图重叠
plt.tight_layout()

# 保存为PDF格式并指定路径
save_path = "/public/ly/zyt/xianyu/CKA_visualization_combined.pdf"
plt.savefig(save_path, format='pdf')

# 显示图形
plt.show()
