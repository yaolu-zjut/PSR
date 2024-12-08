import matplotlib.pyplot as plt
import numpy as np
# 设置字体为新罗马字体
# plt.rcParams['font.family'] = 'Times New Roman'

# 全局设置字体大小
plt.rcParams['font.size'] = 14  # 这里的14是你想要的字体大小
plt.rcParams['legend.fontsize'] = 18  # 设置全局图例字体大小
# 创建一个新的图形并设置大小
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# 第一个子图
x = list(range(10))
y = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]
colors = ['lightblue' if i not in [1, 3, 5, 8] else 'darkblue' for i in range(10)]

bars = axs[0].bar(x, y, color=colors)
# axs[0].set_title('vgg16 + radio128')
axs[0].set_xlabel('Layer ID', fontsize=18)
axs[0].set_ylabel('Number of channels', fontsize=18)
axs[0].set_xticks(x)
axs[0].set_xticklabels(x)

axs[0].set_yticks(np.linspace(0, 600, 6))

# 标注每个柱子上的值
for i, bar in enumerate(bars):
    axs[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(y[i]), ha='center', va='bottom')

# 第二个子图
epochs = range(1, 51)
test_acc_history1 = [0.7323, 0.7754, 0.7942, 0.7976, 0.7949, 0.8086, 0.8026, 0.8096, 0.7981, 0.8131, 0.8068, 0.8152, 0.8169, 0.8269, 0.8131, 0.8139, 0.8153, 0.8274, 0.8149, 0.8183, 0.8302, 0.8071, 0.8218, 0.8304, 0.8236, 0.8304, 0.8173, 0.8252, 0.8298, 0.8219, 0.8133, 0.8191, 0.8347, 0.8312, 0.8258, 0.8189, 0.8291, 0.8316, 0.8248, 0.8186, 0.8338, 0.8300, 0.8363, 0.8359, 0.8424, 0.8269, 0.8372, 0.8397, 0.8330, 0.8386]
test_acc_history2 = [0.6738, 0.7154, 0.7168, 0.7406, 0.7590, 0.7616, 0.7568, 0.7785, 0.7868, 0.7907, 0.7917, 0.8015, 0.7919, 0.7819, 0.7995, 0.7881, 0.7833, 0.7980, 0.7941, 0.7888, 0.8001, 0.8036, 0.8043, 0.7973, 0.7924, 0.8060, 0.8045, 0.7989, 0.8021, 0.8059, 0.8048, 0.8038, 0.8078, 0.8076, 0.7905, 0.8020, 0.8135, 0.8110, 0.8114, 0.8066, 0.8099, 0.8084, 0.8167, 0.8119, 0.8065, 0.8127, 0.8201, 0.8139, 0.8072, 0.8168]

axs[1].plot(epochs, test_acc_history1, label='fine-tuning')
axs[1].plot(epochs, test_acc_history2, label='training from scratch')
# 设置图例的位置为右下角
axs[1].legend(loc='lower right')
# axs[1].set_title('legend')
axs[1].set_xlabel('Epochs', fontsize=18)
axs[1].set_ylabel('Test Accuracy', fontsize=18)


# 设置y轴的刻度数量为6
axs[1].set_yticks(np.linspace(0.65, 0.85, 6))

axs[1].legend()

# 去掉多余白边
plt.tight_layout()
# 保存为PDF格式并指定路径
save_path = "/public/ly/zyt/xianyu/vgg16+radio128_combined.pdf"
plt.savefig(save_path, format='pdf')

# 显示图形
plt.show()
