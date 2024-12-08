import datetime
import os
import sys
import time
import copy
import torch
import torch.nn.init as init
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch import nn
from args import args
from torch.optim import lr_scheduler
from utils1.utils import set_gpu, get_logger, Logger, set_random_seed
from utils1.get_model import get_model
from model_signal.resnet import ResNet_signal, ResNet56_signal_KD
from utils1.get_params import get_cresnet_layer_params, load_cresnet_layer_params

'''

python finetune_or_random.py --gpu 3 --arch ResNet56_signal_KD --set radio1024 --batch_size 128 --weight_decay 0.005 --epochs 50 --lr 0.001 --finetune

resnet56/radio1024/k=3/pruning_rate=89%/test acc
'''
def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    main_worker(args)

def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('pretrained_model/' + args.arch + '/' + args.set):
        os.makedirs('pretrained_model/' + args.arch + '/' + args.set, exist_ok=True)
    logger = get_logger('pretrained_model/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args.arch)
    logger.info(args.set)
    logger.info(args.batch_size)
    logger.info(args.weight_decay)
    logger.info(args.lr)
    logger.info(args.epochs)
    logger.info(args.lr_decay_step)
    logger.info(args.num_classes)

    model_s = get_model(args)
    model_s = set_gpu(args, model_s)
    print(model_s)
    logger.info(model_s)

    if args.arch == 'ResNet56_signal_KD':
        model = ResNet56_signal_KD(dataset=args.set).cuda()
        if args.finetune:
            orginal_model = ResNet_signal(depth=56)
            if args.set == 'radio1024':
                remain_list = [[0], [8], [8]]   # 修改
            model1 = random_initialize_model(orginal_model)
            orginal_state_list = get_cresnet_layer_params(model1)
            model = load_cresnet_layer_params2(orginal_state_list, model, remain_list, num_of_block=9)

    criterion = nn.CrossEntropyLoss().cuda()
    batch_signal, data_sizes = prepare_data(args)  # 得到批训练集和批测试集
    optimizer_ft1 = optim.Adam(filter(lambda p: p.requires_grad, model_s.parameters()), lr=args.lr)
    optimizer_ft2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # 学习率衰减
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer_ft1, step_size=10, gamma=0.8)
    exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer_ft2, step_size=10, gamma=0.8)
    # 训练模型
    test_acc_history2 = train_model(args, batch_signal, data_sizes, model, criterion, optimizer_ft2, exp_lr_scheduler2, num_epochs=args.epochs)
    test_acc_history1 = train_model(args, batch_signal, data_sizes, model_s, criterion, optimizer_ft1, exp_lr_scheduler1, num_epochs=args.epochs)


# 假设 test_acc_history1 和 test_acc_history2 是两个模型的每轮测试准确率历史数组

    epochs = range(1, args.epochs + 1)  # 生成1到50的范围作为横坐标

    # 绘制第一个模型的测试准确率曲线
    plt.plot(epochs, test_acc_history1, label='Model 1')

    # 绘制第二个模型的测试准确率曲线
    plt.plot(epochs, test_acc_history2, label='Model 2')

    plt.title('legend')
    plt.xlabel('Epochs')
    plt.ylabel('Train Accuracy')
    plt.legend()  # 显示图例
    # 保存为PDF文件
    plt.savefig('/public/ly/zyt/xianyu/test_accuracy_plot.pdf')
    plt.show()

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        # 使用正态分布随机初始化权重
        init.kaiming_normal_(module.weight)
        # 初始化偏置为0
        if module.bias is not None:
            init.constant_(module.bias, 0)

def random_initialize_model(orginal_model):
    orginal_model.conv1.apply(initialize_weights)
    orginal_model.bn1.apply(initialize_weights)
    for child in orginal_model.layer1.children():
        child.apply(initialize_weights)
    for child in orginal_model.layer2.children():
        child.apply(initialize_weights)
    for child in orginal_model.layer3.children():
        child.apply(initialize_weights)
    orginal_model.fc.apply(initialize_weights)
    return orginal_model
def load_cresnet_layer_params2(orginal_state_list, pruned_model, remain_layer_list, num_of_block):
    i = 0
    j = 0
    k = 0
    pruned_model.conv1.load_state_dict(orginal_state_list[0])
    pruned_model.bn1.load_state_dict(orginal_state_list[1])
    for child in pruned_model.layer1.children():
        child.load_state_dict(orginal_state_list[remain_layer_list[0][i] + 2])
        i += 1
    for child in pruned_model.layer2.children():
        if j == 0:
            if remain_layer_list[1][j] != 0:
                print('The first layer in layer 2 is initialized by random')
            else:
                child.load_state_dict(orginal_state_list[remain_layer_list[1][j] + num_of_block + 2])
        else:
            child.load_state_dict(orginal_state_list[remain_layer_list[1][j] + num_of_block + 2])
        j += 1
    for child in pruned_model.layer3.children():
        if k == 0:
            if remain_layer_list[2][k] != 0:
                print('The first layer in layer 3 is initialized by random')
            else:
                child.load_state_dict(orginal_state_list[remain_layer_list[2][k] + 2 * num_of_block + 2])
        else:
            child.load_state_dict(orginal_state_list[remain_layer_list[2][k] + 2 * num_of_block + 2])
        k += 1
    pruned_model.fc.load_state_dict(orginal_state_list[-1])
    return pruned_model
def train_model(args, batch_signal, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()  # 开始训练的时间
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    test_acc_history = []  # 用于保存每轮测试的准确率

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

            if phase == 'test':
                test_acc_history.append(epoch_acc.item())

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # 保存当前的训练集精度、测试集精度和最高测试集精度
            'train'
        # 保存测试精度最高时的模型参数
        # if not os.path.exists('./pretrained_signal/result'):
        #     os.mkdir('./pretrained_signal/result')
        # torch.save(best_model_wts, "./pretrained_signal/result/{}_{}_best_lr={}.pth".format(args.dataset, args.model, args.lr))
        print('Best test Acc: {:4f}'.format(best_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    #
    # torch.save(model.state_dict(), 'pretrained_model/' + args.arch + '/' + args.set + "/K4_{}_{}_86%.pt".format(args.arch, args.set))  # 修改剪枝比75%

    return test_acc_history

def prepare_data(args):
    # 导入数据集 , 自己修改数据集
    if args.set == 'radio128':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/radio128/radio128NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/radio128/radio128NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/radio128/radio128NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/radio128/radio128NormTestSnrY.npy'  # 训练集标签
    elif args.set == 'radio512':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/radio512/radio512NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/radio512/radio512NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/radio512/radio512NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/radio512/radio512NormTestSnrY.npy'  # 训练集标签
    elif args.set == 'radio1024':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTestSnrY.npy'  # 训练集标签
    elif args.set == 'all_radio128':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/alldb/radio128/radio128NormTestSnrY.npy'  # 训练集标签
    elif args.set == 'all_radio512':
        train_2 = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTrainX.npy')
        train_label_path = '/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTrainSnrY.npy'  # 训练集标签
        test_2 = np.load('/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTestX.npy')
        test_label_path = '/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTestSnrY.npy'  # 训练集标签

    if args.set in ['radio128', 'radio512', 'radio1024']:
        train_label = np.load(train_label_path)[:,0]  # 得到0到11的类标签数据
        test_label = np.load(test_label_path)[:,0]  # 得到0到11的类标签数据
    elif args.set in ['all_radio128', 'all_radio512']:
        train_label = np.load(train_label_path)[1]  # 得到0到11的类标签数据
        test_label = np.load(test_label_path)[1]  # 得到0到11的类标签数据
    data_sizes = {'train': len(train_label), 'test': len(test_label)}
    # 数组变张量

    # print('Data will be reshaped into [N, 1, 2, Length]')
    train_2 = torch.from_numpy(train_2)
    train_2 = train_2.permute(0, 2, 1)
    train_2 = torch.unsqueeze(train_2, dim=1)
    train_2 = train_2.type(torch.FloatTensor)
    test_2 = torch.from_numpy(test_2)
    test_2 = test_2.permute(0, 2, 1)  # 交换位置
    test_2 = torch.unsqueeze(test_2, dim=1)  # 增加一个维度
    test_2 = test_2.type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label)
    train_label = train_label.type(torch.LongTensor)
    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)

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
                                                         shuffle=True, num_workers=8),
                    'test': torch.utils.data.DataLoader(dataset=test_signal, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=8)}
    return batch_signal, data_sizes

if __name__ == "__main__":
    main()