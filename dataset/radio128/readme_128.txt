npy为>=10dB归一化信号长度128 类别11类

npy为[10, 12, 14, 16, 18]dB归一化信号
train_num = 200  # 每类每个信噪比取200个做训练集
test_num = 50  # 每类每个信噪比取50个做训练集


radio512NormTrainX.npy  # (44000, 128, 2)
radio512NormTrainSnrY.npy  #  (44000, 2)  (label, snr)

radio512NormTestX.npy  # (11000, 128, 2)
radio512NormTestSnrY.npy  # (11000, 2)  (label, snr)
