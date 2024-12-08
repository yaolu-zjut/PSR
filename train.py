import argparse
import os
import numpy as np
import time
import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from model_signal.vgg16 import vgg16_signal
from model_signal.resnet import ResNet_signal
from model_signal.CNN1D import ResNet1D
from model_signal.signet import ResNet50

'''
# setup up:
python train.py
'''

def train_model(args, batch_signal, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()  # 开始训练的时间
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            signal_num = 0
            # Iterate over data.
            pbar = tqdm(batch_signal[phase])
            for inputs, labels in pbar:
                inputs = inputs.cuda()  # (batch_size, 2, 128)
                labels = labels.cuda()  # (batch_size, )
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if args.model == 'CNN1D' or args.model == 'SigNet50':
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    else:
                        l1_regularization = torch.tensor(0.).cuda()
                        for param in model.parameters():
                            l1_regularization += torch.norm(param, 1)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels) + 0.0001 * l1_regularization

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                signal_num += inputs.size(0)
                # 在进度条的右边实时显示数据集类型、loss值和精度
                epoch_loss = running_loss / signal_num
                epoch_acc = running_corrects.double() / signal_num
                pbar.set_postfix({'Set': '{}'.format(phase),
                                  'Loss': '{:.4f}'.format(epoch_loss),
                                  'Acc': '{:.4f}'.format(epoch_acc)})
                # print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), end=' ')
            if phase == 'train':
                scheduler.step()
            # 显示该轮的loss和精度
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # 保存当前的训练集精度、测试集精度和最高测试集精度
            'train'
        # 保存测试精度最高时的模型参数
        if not os.path.exists('./pretrained_signal/result'):
            os.mkdir('./pretrained_signal/result')
        torch.save(best_model_wts, "./pretrained_signal/result/{}_{}_best_lr={}_{}.pth".format(args.dataset, args.model, args.lr, args.initialize))
        print('Best test Acc: {:4f}'.format(best_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def prepare_data(args):
    # 导入数据集
    if args.dataset == 'radio128':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/radio128/radio128NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/radio128/radio128NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/radio128/radio128NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/radio128/radio128NormTestSnrY.npy'  # 训练集标签
    elif args.dataset == 'radio512':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/radio512/radio512NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/radio512/radio512NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/radio512/radio512NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/radio512/radio512NormTestSnrY.npy'  # 训练集标签
    elif args.dataset == 'radio1024':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTestSnrY.npy'  # 训练集标签
    elif args.dataset == 'all_radio128':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTestSnrY.npy'  # 训练集标签
    elif args.dataset == 'all_radio512':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTestSnrY.npy'  # 训练集标签


    if args.dataset in ['radio128', 'radio512', 'radio1024']:
        train_label = np.load(train_label_path)[:, 0]  # 得到0到11的类标签数据
        test_label = np.load(test_label_path)[:, 0]  # 得到0到11的类标签数据
    elif args.dataset in ['all_radio128', 'all_radio512']:
        train_label = np.load(train_label_path)[1]  # 得到0到11的类标签数据
        test_label = np.load(test_label_path)[1]  # 得到0到11的类标签数据
    train_label = torch.from_numpy(train_label)
    train_label = train_label.type(torch.LongTensor)
    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)

    data_sizes = {'train': len(train_label), 'test': len(test_label)}

    if args.model == 'CNN1D' or args.model == 'SigNet50':
        train_2 = torch.from_numpy(train_2).permute(0, 2, 1)  # [312000, 2, 128] 转为[N, Channel, Length]
        train_2 = train_2.type(torch.FloatTensor)
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)

    else:
        train_2 = torch.from_numpy(train_2)
        train_2 = train_2.permute(0, 2, 1)
        train_2 = torch.unsqueeze(train_2, dim=1)
        train_2 = train_2.type(torch.FloatTensor)
        test_2 = torch.from_numpy(test_2)
        test_2 = test_2.permute(0, 2, 1)  # 交换位置
        test_2 = torch.unsqueeze(test_2, dim=1)  # 增加一个维度
        test_2 = test_2.type(torch.FloatTensor)

    # 训练集标签类别数
    train_num_classes = len(torch.unique(train_label))
    print("Number of classes in train_label:", train_num_classes)
    # 测试集标签类别数
    test_num_classes = len(torch.unique(test_label))
    print("Number of classes in test_label:", test_num_classes)
    print(train_2.shape, train_label.shape, test_2.shape, test_label.shape)
    # 把数据放在数据库中
    train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    # 将训练集和测试集分批
    batch_signal = {'train': torch.utils.data.DataLoader(dataset=train_signal, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.num_workers),
                    'test': torch.utils.data.DataLoader(dataset=test_signal, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=args.num_workers)}

    return batch_signal, data_sizes
import torch.nn.init as init

# 定义初始化函数
def initialize_weights(model, init_type=None):
    """
    初始化模型权重
    :param model: 要初始化的模型
    :param init_type: 初始化类型，"xavier", "random", 或 "lecun"
    """
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):  # 适用于 Conv2d 和 Linear 层
            if init_type == "xavier":
                init.xavier_uniform_(m.weight)
            elif init_type == "random":
                init.uniform_(m.weight, a=-0.1, b=0.1)  # 使用均匀分布随机初始化
            elif init_type == "lecun":
                init.kaiming_uniform_(m.weight, a=1, mode='fan_in', nonlinearity='linear')
            if m.bias is not None:
                init.constant_(m.bias, 0)  # 偏置初始化为 0

def arg_parse():
    parser = argparse.ArgumentParser(description='Signal_diagonal_matrix_CNN arguments.')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, help='Number of workers to load data.')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset: radio128, radio512, radio1024, all_radio128, all_radio512')
    parser.add_argument('--model', dest='model', type=str, help='Model: ResNet56_signal, ResNet110_signal, vgg16_signal, CNN1D, SigNet50')
    parser.add_argument('--initialize', dest='initialize', type=str, help='initialize: xavier, random, lecun')
    parser.set_defaults(lr=0.001, batch_size=128, num_epochs=30, num_workers=8, dataset='all_radio128', model='ResNet56_signal', initialize='xavier')
    return parser.parse_args()
# python train.py --initialize 'xavier'
# python train.py --initialize 'random'
# python train.py --initialize 'lecun'

def main():
    prog_args = arg_parse()
    torch.cuda.set_device(0)
    output_directory = "./pretrained_signal/result/"
    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, "result_{}_{}.csv".format(prog_args.dataset, prog_args.model)), 'a', newline='') as t:
        writer_train = csv.writer(t)
        writer_train.writerow(['dataset={}, model={}, num_epoch={}, lr={}, batch_size={}'.format(prog_args.dataset, prog_args.model, prog_args.num_epochs, prog_args.lr, prog_args.batch_size)])
        writer_train.writerow(['epoch', 'phase', 'epoch_loss', 'epoch_acc', 'best_acc'])
    batch_signal, data_sizes = prepare_data(prog_args)  # 跳转到prepare_data函数，得到批训练集和批测试集
    # 模型放到GPU上
    if prog_args.model == 'ResNet56_signal':
        model_ft = ResNet_signal(depth=56, dataset=prog_args.dataset).cuda()
    elif prog_args.model == 'ResNet110_signal':
        model_ft = ResNet_signal(depth=110, dataset=prog_args.dataset).cuda()
    elif prog_args.model == 'vgg16_signal':
        model_ft = vgg16_signal(dataset=prog_args.dataset).cuda()
    elif prog_args.model == 'CNN1D':
        model_ft = ResNet1D(dataset=prog_args.dataset).cuda()
    elif prog_args.model == 'SigNet50':
        model_ft = ResNet50(dataset=prog_args.dataset).cuda()
    else:
        print('Error! There is no model of your choice')
    print(model_ft)
    # 导入预训练模型
    # model_ft.load_state_dict(torch.load(r'D:\program_Q\new\torchvision_model\resnet50-19c8e357.pth'))
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    # optimizer_ft = optim.SGD([{"params": model_ft.parameters()}, {"params": filter_all}, {"params": bias_all}],
    #                          lr=prog_args.lr, momentum=0.9)

    # 选择初始化方式：xavier, random, lecun
    initialize_weights(model_ft, init_type=prog_args.initialize)  # 使用 Xavier 初始化
    # initialize_weights(model_ft, init_type="he")   # 使用 He 初始化
    # initialize_weights(model_ft, init_type="lecun") # 使用 LeCun 初始化

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=prog_args.lr)
    # 学习率衰减
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.8)
    # 训练模型
    train_model(prog_args, batch_signal, data_sizes, model_ft, criterion, optimizer_ft,
                exp_lr_scheduler, num_epochs=prog_args.num_epochs)


if __name__ == '__main__':
    main()
