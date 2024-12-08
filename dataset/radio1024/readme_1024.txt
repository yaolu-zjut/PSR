原radioSignalRML2018_OSC数据集
f = h5py.File(DATA_PATH+FILE_NAME, 'r')  # 打开h5文件
print(f.keys())  # 可以查看所有的主键
x = f['X'][:]  # (2555904, 1024, 2)  # 长度1024
y = f['Y'][:]  # (2555904, 24)  # one-hot类别 共24类
z = f['Z'][:]  # (2555904, 1) #  SNR [-20, -18, ..., 30] 2dB间隔 共26种信噪比 每类每个信噪比4096个样本

npy为[10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]dB归一化信号
train_num = 3200  # 每类每个信噪比取3200个做训练集
test_num = 800  # 每类每个信噪比取800个做训练集

radio1024NormTrainX.npy  # (844800, 1024, 2)
radio1024NormTrainSnrY.npy  #  (844800, 2)  (label, snr)

radio1024NormTestX.npy  # (211200, 1024, 2)
radio1024NormTestSnrY.npy  # (211200, 2)  (label, snr)
