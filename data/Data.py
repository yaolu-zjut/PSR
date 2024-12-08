import torch
from torch.utils.data.sampler import SubsetRandomSampler
from args import args
import numpy as np
from torch.utils.data import TensorDataset

class radio128:
    def __init__(self):
        super(radio128, self).__init__()

        x_test = np.load('/public/ly/zyt/xianyu/dataset/radio128/radio128NormTestX.npy')
        y_test = np.load('/public/ly/zyt/xianyu/dataset/radio128/radio128NormTestSnrY.npy')
        x_train = np.load('/public/ly/zyt/xianyu/dataset/radio128/radio128NormTrainX.npy')
        y_train = np.load('/public/ly/zyt/xianyu/dataset/radio128/radio128NormTrainSnrY.npy')
        y_train = y_train[:,0]
        y_test = y_test[:,0]
        self.dataset_sizes = {'train': len(y_train), 'test': len(y_test)}
        # 查看唯一值
        # unique_labels_train = np.unique(y_train)
        # unique_labels_test = np.unique(y_test)
        # print("Unique labels in y_train:", unique_labels_train)
        # print("Number of unique labels in y_train:", len(unique_labels_train))
        # print("Unique labels in y_test:", unique_labels_test)
        # print("Number of unique labels in y_test:", len(unique_labels_test))

        # 划分数据集
        x_train = torch.from_numpy(x_train)
        x_train = x_train.permute(0, 2, 1)
        x_train = torch.unsqueeze(x_train, dim=1)
        x_train = x_train.type(torch.FloatTensor)
        x_test = torch.from_numpy(x_test)
        x_test = x_test.permute(0, 2, 1)  # 交换位置
        x_test = torch.unsqueeze(x_test, dim=1)  # 增加一个维度
        x_test = x_test.type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train)
        y_train = y_train.type(torch.LongTensor)
        y_test = torch.from_numpy(y_test)
        y_test = y_test.type(torch.LongTensor)
        # 转换数据为 DataLoader
        # dataset_train = TensorDataset(x_train, y_train)
        # dataset_test = TensorDataset(x_test, y_test)
        dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
        dataset_test = torch.utils.data.TensorDataset(x_test, y_test)

        # # 训练集标签类别数
        # train_num_classes = len(torch.unique(y_train))
        # print("Number of classes in train_label:", train_num_classes)
        # # 测试集标签类别数
        # test_num_classes = len(torch.unique(y_test))
        # print("Number of classes in test_label:", test_num_classes)
        # print(y_train.shape)  # torch.Size([44000]) 44000是数量,2是通道,128是长度,1是增加的一个维度
        # print(x_train.shape)  # torch.Size([44000, 1, 2, 128])
        # print(y_test.shape)  # torch.Size([11000])
        # print(x_test.shape)  # torch.Size([11000, 1, 2, 128])
        self.train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
        self.val_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

class radio512:
    def __init__(self):
        super(radio512, self).__init__()

        x_test = np.load('/public/ly/zyt/xianyu/dataset/radio512/radio512NormTestX.npy')
        y_test = np.load('/public/ly/zyt/xianyu/dataset/radio512/radio512NormTestSnrY.npy')
        x_train = np.load('/public/ly/zyt/xianyu/dataset/radio512/radio512NormTrainX.npy')
        y_train = np.load('/public/ly/zyt/xianyu/dataset/radio512/radio512NormTrainSnrY.npy')
        y_train = y_train[:,0]
        y_test = y_test[:,0]
        self.dataset_sizes = {'train': len(y_train), 'test': len(y_test)}

        # 划分数据集
        x_train = torch.from_numpy(x_train)
        x_train = x_train.permute(0, 2, 1)
        x_train = torch.unsqueeze(x_train, dim=1)
        x_train = x_train.type(torch.FloatTensor)
        x_test = torch.from_numpy(x_test)
        x_test = x_test.permute(0, 2, 1)  # 交换位置
        x_test = torch.unsqueeze(x_test, dim=1)  # 增加一个维度
        x_test = x_test.type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train)
        y_train = y_train.type(torch.LongTensor)
        y_test = torch.from_numpy(y_test)
        y_test = y_test.type(torch.LongTensor)
        # 转换数据为 DataLoader
        # dataset_train = TensorDataset(x_train, y_train)
        # dataset_test = TensorDataset(x_test, y_test)
        dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
        dataset_test = torch.utils.data.TensorDataset(x_test, y_test)

        # print(y_train.shape)  # torch.Size([132000]) 132000是数量,2是通道,512是长度,1是增加的一个维度
        # print(x_train.shape)  # torch.Size([132000, 1, 2, 512])
        # print(y_test.shape)  # torch.Size([66000])
        # print(x_test.shape)  # torch.Size([66000, 1, 2, 512])
        self.train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
        self.val_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

class radio1024:
    def __init__(self):
        super(radio1024, self).__init__()

        x_test = np.load('/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTestX.npy')
        y_test = np.load('/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTestSnrY.npy')
        x_train = np.load('/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy')
        y_train = np.load('/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy')
        y_train = y_train[:,0]
        y_test = y_test[:,0]
        self.dataset_sizes = {'train': len(y_train), 'test': len(y_test)}

        # 划分数据集
        if args.arch in ['CNN1D_KD', 'CNN1D', 'SigNet50', 'SigNet50_KD']:
            x_train = torch.from_numpy(x_train).permute(0, 2, 1)  # [312000, 2, 128] 转为[N, Channel, Length]
            x_train = x_train.type(torch.FloatTensor)
            x_test = torch.from_numpy(x_test).permute(0, 2, 1)  # [156000, 2, 128]
            x_test = x_test.type(torch.FloatTensor)
        else:
            # 划分数据集
            x_train = torch.from_numpy(x_train)
            x_train = x_train.permute(0, 2, 1)
            x_train = torch.unsqueeze(x_train, dim=1)
            x_train = x_train.type(torch.FloatTensor)
            x_test = torch.from_numpy(x_test)
            x_test = x_test.permute(0, 2, 1)  # 交换位置
            x_test = torch.unsqueeze(x_test, dim=1)  # 增加一个维度
            x_test = x_test.type(torch.FloatTensor)

        y_train = torch.from_numpy(y_train)
        y_train = y_train.type(torch.LongTensor)
        y_test = torch.from_numpy(y_test)
        y_test = y_test.type(torch.LongTensor)
        # 转换数据为 DataLoader
        # dataset_train = TensorDataset(x_train, y_train)
        # dataset_test = TensorDataset(x_test, y_test)
        dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
        dataset_test = torch.utils.data.TensorDataset(x_test, y_test)

        # print(y_train.shape)  # torch.Size([844800]) 132000是数量,2是通道,1024是长度,1是增加的一个维度
        # print(x_train.shape)  # torch.Size([844800, 1, 2, 1024])
        # print(y_test.shape)  # torch.Size([211200])
        # print(x_test.shape)  # torch.Size([211200, 1, 2, 1024])
        self.train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
        self.val_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

class all_radio128:
    def __init__(self):
        super(all_radio128, self).__init__()

        x_test = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTestX.npy')
        y_test = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTestSnrY.npy')
        x_train = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTrainX.npy')
        y_train = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTrainSnrY.npy')
        y_train = y_train[1]
        y_test = y_test[1]
        self.dataset_sizes = {'train': len(y_train), 'test': len(y_test)}
        # 查看唯一值
        # unique_labels_train = np.unique(y_train)
        # unique_labels_test = np.unique(y_test)
        # print("Unique labels in y_train:", unique_labels_train)
        # print("Number of unique labels in y_train:", len(unique_labels_train))
        # print("Unique labels in y_test:", unique_labels_test)
        # print("Number of unique labels in y_test:", len(unique_labels_test))

        if args.arch in ['CNN1D_KD', 'CNN1D', 'SigNet50', 'SigNet50_KD']:
            x_train = torch.from_numpy(x_train).permute(0, 2, 1)  # [312000, 2, 128] 转为[N, Channel, Length]
            x_train = x_train.type(torch.FloatTensor)
            x_test = torch.from_numpy(x_test).permute(0, 2, 1)  # [156000, 2, 128]
            x_test = x_test.type(torch.FloatTensor)
        else:
            # 划分数据集
            x_train = torch.from_numpy(x_train)
            x_train = x_train.permute(0, 2, 1)
            x_train = torch.unsqueeze(x_train, dim=1)
            x_train = x_train.type(torch.FloatTensor)
            x_test = torch.from_numpy(x_test)
            x_test = x_test.permute(0, 2, 1)  # 交换位置
            x_test = torch.unsqueeze(x_test, dim=1)  # 增加一个维度
            x_test = x_test.type(torch.FloatTensor)

        y_train = torch.from_numpy(y_train)
        y_train = y_train.type(torch.LongTensor)
        y_test = torch.from_numpy(y_test)
        y_test = y_test.type(torch.LongTensor)
        # 转换数据为 DataLoader
        # dataset_train = TensorDataset(x_train, y_train)
        # dataset_test = TensorDataset(x_test, y_test)
        dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
        dataset_test = torch.utils.data.TensorDataset(x_test, y_test)

        # # 训练集标签类别数
        # train_num_classes = len(torch.unique(y_train))
        # print("Number of classes in train_label:", train_num_classes)
        # # 测试集标签类别数
        # test_num_classes = len(torch.unique(y_test))
        # print("Number of classes in test_label:", test_num_classes)
        # print(y_train.shape)  # torch.Size([176000]) 44000是数量,2是通道,128是长度,1是增加的一个维度
        # print(x_train.shape)  # torch.Size([176000, 1, 2, 128])
        # print(y_test.shape)  # torch.Size([44000])
        # print(x_test.shape)  # torch.Size([44000, 1, 2, 128])
        self.train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
        self.val_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

class all_radio512:
    def __init__(self):
        super(all_radio512, self).__init__()

        x_test = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTestX.npy')
        y_test = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTestSnrY.npy')
        x_train = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTrainX.npy')
        y_train = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTrainSnrY.npy')
        y_train = y_train[1]
        y_test = y_test[1]
        self.dataset_sizes = {'train': len(y_train), 'test': len(y_test)}

        if args.arch in ['CNN1D_KD', 'CNN1D', 'SigNet50', 'SigNet50_KD']:
            x_train = torch.from_numpy(x_train).permute(0, 2, 1)  # [312000, 2, 128] 转为[N, Channel, Length]
            x_train = x_train.type(torch.FloatTensor)
            x_test = torch.from_numpy(x_test).permute(0, 2, 1)  # [156000, 2, 128]
            x_test = x_test.type(torch.FloatTensor)
        else:
            # 划分数据集
            x_train = torch.from_numpy(x_train)
            x_train = x_train.permute(0, 2, 1)
            x_train = torch.unsqueeze(x_train, dim=1)
            x_train = x_train.type(torch.FloatTensor)
            x_test = torch.from_numpy(x_test)
            x_test = x_test.permute(0, 2, 1)  # 交换位置
            x_test = torch.unsqueeze(x_test, dim=1)  # 增加一个维度
            x_test = x_test.type(torch.FloatTensor)

        y_train = torch.from_numpy(y_train)
        y_train = y_train.type(torch.LongTensor)
        y_test = torch.from_numpy(y_test)
        y_test = y_test.type(torch.LongTensor)
        # 转换数据为 DataLoader
        # dataset_train = TensorDataset(x_train, y_train)
        # dataset_test = TensorDataset(x_test, y_test)
        dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
        dataset_test = torch.utils.data.TensorDataset(x_test, y_test)

        # print(y_train.shape)  # torch.Size([312000]) 132000是数量,2是通道,512是长度,1是增加的一个维度
        # print(x_train.shape)  # torch.Size([312000, 1, 2, 512])
        # print(y_test.shape)  # torch.Size([156000])
        # print(x_test.shape)  # torch.Size([156000, 1, 2, 512])
        self.train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
        self.val_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8)