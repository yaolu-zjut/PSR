import datetime
import os
import time
import torch
from itertools import combinations
from args import args
from model_signal.vgg16 import vgg16_signal
from model_signal.resnet import ResNet_signal
from model_signal.CNN1D import ResNet1D
from model_signal.signet import ResNet50
from utils1.get_dataset import get_dataset
from utils1.get_model import get_model
from utils1.get_params import get_layer_params
from utils1.utils import set_random_seed, set_gpu, get_logger
from zero_nas import ZeroNas

'''
设置每块需要保留几层 剪枝比50%和75%左右，75%：7，21，42，50%：5，14，27, 25%:3,7,14

python Reassembly.py --gpu 4 --arch vgg16_signal --set radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch vgg16_signal --set radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch vgg16_signal --set radio1024 --num_classes 24 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch ResNet56_signal --set radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch ResNet56_signal --set radio1024 --num_classes 24 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch ResNet110_signal --set radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch ResNet110_signal --set radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 0 --arch ResNet110_signal --set radio1024 --num_classes 24 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow

python Reassembly.py --gpu 4 --arch vgg16_signal --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch vgg16_signal --set all_radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow

python Reassembly.py --gpu 4 --arch ResNet56_signal --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow

python Reassembly.py --gpu 4 --arch ResNet56_signal --set all_radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch ResNet110_signal --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch ResNet110_signal --set all_radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow

信号专用模型：
python Reassembly.py --gpu 4 --arch CNN1D --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch CNN1D --set all_radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch CNN1D --set radio1024 --num_classes 24 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 4 --arch SigNet50 --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
python Reassembly.py --gpu 1 --arch SigNet50 --set all_radio512 --num_classes 12 --batch_size 32 --pretrained --evaluate  --zero_proxy synflow
'''
# 最大分类数 56: k = 5, 110: k = 7
partition_vgg16_signal_radio128 = [1, 1, 1, 2, 2, 2, 2, 3, 3]
partition_vgg16_signal_radio512 = [1, 2, 2, 2, 2, 2, 2, 3, 3]
partition_vgg16_signal_radio1024 = [1, 2, 2, 2, 2, 2, 3, 3, 3]
partition_ResNet56_signal_radio128 = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

# K=3:1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
# k=5:1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5
# k=7:1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7
# K=9:1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9
# k=11:1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 11
# k=13:1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13

cosine_ResNet56_signal_radio128 = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
partition_ResNet56_signal_radio512 = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3]
partition_ResNet56_signal_radio1024 = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3]
partition_ResNet110_signal_radio128 = [1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
partition_ResNet110_signal_radio512 = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
partition_ResNet110_signal_radio1024 = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
partition_vgg16_signal_all_radio128 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
partition_vgg16_signal_all_radio512 = [1, 1, 2, 2, 2, 2, 2, 3, 3]

partition_ResNet56_signal_all_radio128 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]

partition_ResNet56_signal_all_radio512 = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
partition_ResNet110_signal_all_radio128 = [1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
partition_ResNet110_signal_all_radio512 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

partition_CNN1D_all_radio128 = [1, 1, 1, 1, 2, 2]
partition_CNN1D_all_radio512 = [1, 1, 1, 1, 2, 2]
partition_CNN1D_radio1024 = [1, 1, 2, 2, 2, 2]
partition_SigNet50_all_radio128 = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
partition_SigNet50_all_radio512 = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
layer_n = {'ResNet56_signal': 28, 'ResNet110_signal': 55, 'vgg16_signal': 9, 'SigNet50':16}
vgg16_signal_layer_name = ['layer1.1.0', 'layer2.0.0', 'layer2.1.0', 'layer3.0.0', 'layer3.1.0', 'layer3.2.0', 'layer4.0.0', 'layer4.1.0', 'layer4.2.0']
CNN1D_layer_name = ['conv1.conv1', 'conv1.conv2', 'conv1.conv3', 'conv1.conv4', 'conv1.conv5', 'conv2.conv1', 'conv2.conv2', 'conv2.conv3', 'conv2.conv4', 'conv2.conv5',
                    'conv3.conv1', 'conv3.conv2', 'conv3.conv3', 'conv3.conv4', 'conv3.conv5', 'conv4.conv1', 'conv4.conv2', 'conv4.conv3', 'conv4.conv4', 'conv4.conv5',
                    'conv5.conv1', 'conv5.conv2', 'conv5.conv3', 'conv5.conv4', 'conv5.conv5', 'conv6.conv1', 'conv6.conv2', 'conv6.conv3', 'conv6.conv4', 'conv6.conv5']
def get_layer(partition):
    max_part = max(partition)
    print("Max partition: {}".format(max_part))
    block = []
    for i in range(max_part):
        block.append([])
    for i in range(len(partition)):
        for j in range(max_part):
            if partition[i] == j+1:
                block[j].append(i)
    return block, max_part

def main():
    start_time = time.time()  # 记录开始时间
    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set):
        os.makedirs('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set, exist_ok=True)
    logger = get_logger('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set + '/logger' + now + '.log')
    logger.info(args)
    model = get_model(args)
    model = set_gpu(args, model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)
    model.eval()
    # if args.evaluate:
    #     if args.set in ['cifar10', 'cifar100']:
    #         acc1, acc5 = validate(data.val_loader, model, criterion, args)
    #     else:
    #         acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)
    #
    # logger.info(acc1)
    # logger.info(model)

    # 设置每块需要保留几层 ，25%：7，21，41，50%：5，14，27
    if args.arch == 'vgg16_signal':
        if args.set == 'radio128':
            remain_layer = [2, 1, 1]  # 4，
        elif args.set == 'radio512':
            remain_layer = [1, 4, 1]  # 3，
        elif args.set == 'radio1024':
            remain_layer = [1, 3, 1]  # 3，
        elif args.set == 'all_radio128':
            remain_layer = [1, 1, 1]  # 3，需要调整Remaining layers: (1,)(4,)(8,)
        elif args.set == 'all_radio512':
            remain_layer = [1, 1, 1]  # 3，需要调整Remaining layers: (1,)(4,)(7,)
    elif args.arch == 'ResNet56_signal':
        if args.set == 'radio128':
            remain_layer = [1, 9, 4]  # 21,14,7,3
            # K=3:[1, 8, 5]
            # k=5:[1, 1, 3, 7, 2]
            # k=7:[1, 2, 1, 1, 4, 3, 2]
            # K=9:[2, 2, 1, 1, 3, 1, 1, 1, 2]
            # k=11:[2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2]
            # k=13:[1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
        elif args.set == 'radio512':
            remain_layer = [1, 2, 1]  # 21,14,7,4
        elif args.set == 'radio1024':
            remain_layer = [1, 1, 1]  # 21,14,7,3
        elif args.set == 'all_radio128':
            remain_layer = [1, 1, 1, 1]  # 21
        elif args.set == 'all_radio512':
            remain_layer = [1, 2, 2, 1, 1]  # 21
    elif args.arch == 'ResNet110_signal':
        if args.set == 'radio128':
            remain_layer = [1, 1, 1, 2, 1, 1]  # 7:42,27,14,7,6,5,4
        elif args.set == 'radio512':
            remain_layer = [1, 2, 1]  # 6:42,27,14,6
        elif args.set == 'radio1024':
            remain_layer = [2, 4, 12, 4, 3, 2, 15]  # 6:42,27,14,
        elif args.set == 'all_radio128':
            remain_layer = [2, 2, 10, 2, 12, 4, 10]  # 7，需要调整
        elif args.set == 'all_radio512':
            remain_layer = [1, 2, 1]  # 7，需要调整

    elif args.arch == 'CNN1D':
        if args.set == 'all_radio128':
            remain_layer = [1, 1]  # 7，需要调整
        elif args.set == 'all_radio512':
            remain_layer = [1, 1]  # 7，需要调整
        elif args.set == 'radio1024':
            remain_layer = [1, 1]  # 6:42,27,14,
    elif args.arch == 'SigNet50':
        if args.set == 'all_radio128':
            remain_layer = [1, 2, 1, 1]  # 7，需要调整
        elif args.set == 'all_radio512':
            remain_layer = [1, 1, 1, 1]  # 7，需要调整


    value_list = replace_layer_initialization(data, args, remain_layer)
    print(value_list)
    search_best(value_list)

    end_time = time.time()  # 记录结束时间
    elapsed_time = (end_time - start_time) / 60  # 转换为分钟
    print(f"代码运行时间: {elapsed_time:.2f} 分钟")
    # torch.save(value_list, "value_rank_block_0_remain_2_layer_{}_{}.pth".format(args.arch, args.set))
def replace_layer_initialization(data, args, remain_layer):
    """ Run the methods on the data and then saves it to out_path. """
    if args.arch == 'vgg16_signal':
        model1 = vgg16_signal()
        model1 = set_gpu(args, model1)
        model2 = vgg16_signal()
        model2 = set_gpu(args, model2)
        if args.set == 'radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio128_vgg16_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio512_vgg16_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio1024':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_vgg16_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_vgg16_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_vgg16_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
    elif args.arch == 'ResNet56_signal':
        model1 = ResNet_signal(depth=56)
        model1 = set_gpu(args, model1)
        model2 = ResNet_signal(depth=56)
        model2 = set_gpu(args, model2)
        if args.set == 'radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio128_ResNet56_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio512_ResNet56_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio1024':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_ResNet56_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_ResNet56_signal_best_lr=0.001_lecun.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_ResNet56_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
    elif args.arch == 'ResNet110_signal':
        model1 = ResNet_signal(depth=110)
        model1 = set_gpu(args, model1)
        model2 = ResNet_signal(depth=110)
        model2 = set_gpu(args, model2)
        if args.set == 'radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio128_ResNet110_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio512_ResNet110_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio1024':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_ResNet110_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_ResNet110_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_ResNet110_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
    elif args.arch == 'CNN1D':
        model1 = ResNet1D()
        model1 = set_gpu(args, model1)
        model2 = ResNet1D()
        model2 = set_gpu(args, model2)
        if args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_CNN1D_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_CNN1D_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio1024':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/radio1024_CNN1D_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
    elif args.arch == 'SigNet50':
        model1 = ResNet50()
        model1 = set_gpu(args, model1)
        model2 = ResNet50()
        model2 = set_gpu(args, model2)
        if args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_SigNet50_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_SigNet50_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)




    model1.load_state_dict(ckpt)
    model1.eval()
    model2.eval()
    # correct1 = validate_signal_data(data.val_loader, model1, criterion, data.dataset_sizes)
    # correct2 = validate_signal_data(data.val_loader, model2, criterion, data.dataset_sizes)

    if args.arch == 'vgg16_signal':
        if args.set == 'radio128':
            block, max_part = get_layer(partition_vgg16_signal_radio128)
        elif args.set == 'radio512':
            block, max_part = get_layer(partition_vgg16_signal_radio512)
        elif args.set == 'radio1024':
            block, max_part = get_layer(partition_vgg16_signal_radio1024)
        elif args.set == 'all_radio128':
            block, max_part = get_layer(partition_vgg16_signal_all_radio128)
        elif args.set == 'all_radio512':
            block, max_part = get_layer(partition_vgg16_signal_all_radio512)
    elif args.arch == 'ResNet56_signal':
        if args.set == 'radio128':
            block, max_part = get_layer(partition_ResNet56_signal_radio128)
            # block, max_part = get_layer(cosine_ResNet56_signal_radio128)
        elif args.set == 'radio512':
            block, max_part = get_layer(partition_ResNet56_signal_radio512)
        elif args.set == 'radio1024':
            block, max_part = get_layer(partition_ResNet56_signal_radio1024)
        elif args.set == 'all_radio128':
            block, max_part = get_layer(partition_ResNet56_signal_all_radio128)
        elif args.set == 'all_radio512':
            block, max_part = get_layer(partition_ResNet56_signal_all_radio512)
    elif args.arch == 'ResNet110_signal':
        if args.set == 'radio128':
            block, max_part = get_layer(partition_ResNet110_signal_radio128)
        elif args.set == 'radio512':
            block, max_part = get_layer(partition_ResNet110_signal_radio512)
        elif args.set == 'radio1024':
            block, max_part = get_layer(partition_ResNet110_signal_radio1024)
        elif args.set == 'all_radio128':
            block, max_part = get_layer(partition_ResNet110_signal_all_radio128)
        elif args.set == 'all_radio512':
            block, max_part = get_layer(partition_ResNet110_signal_all_radio512)
    elif args.arch == 'CNN1D':
        if args.set == 'all_radio128':
            block, max_part = get_layer(partition_CNN1D_all_radio128)
        elif args.set == 'all_radio512':
            block, max_part = get_layer(partition_CNN1D_all_radio512)
        elif args.set == 'radio1024':
            block, max_part = get_layer(partition_CNN1D_radio1024)
    elif args.arch == 'SigNet50':
        if args.set == 'all_radio128':
            block, max_part = get_layer(partition_SigNet50_all_radio128)
        elif args.set == 'all_radio512':
            block, max_part = get_layer(partition_SigNet50_all_radio512)

    value_list = [[] for i in range(max_part)]

    if args.arch in ['ResNet56_signal', 'ResNet110_signal']:
        random_state_list = get_layer_params(model1)  # 取出预训练好的model1模型参数，将其加载到未经过训练的model2上
        print(block)
        print(max_part)
        for iii in range(max_part):
            for p in combinations(block[iii], remain_layer[iii]):  # remain 3 pretrained layers each part, 第一个部分第一层不删，所以少保留一个
                print(p)
                x = (layer_n[args.arch] - 1) // 3  # ResNet56_signal: x = 9，对于ResNet110_signal: x = 18，对于vgg16_signal: x = 2
                for i in p:  # i 将依次取 0、1 和 2
                    if i == 0:
                        model2.conv1.load_state_dict(random_state_list[0])
                        model2.bn1.load_state_dict(random_state_list[1])
                    elif 0 < i <= x:
                        model2.layer1[i - 1].load_state_dict(random_state_list[i + 1])
                        # print(model2.layer1[i - 1])
                    elif x < i <= 2 * x:
                        model2.layer2[i - 1 - x].load_state_dict(random_state_list[i + 1])
                        # print(model2.layer2[i - 1 - x])
                    elif 2 * x < i <= 3 * x:
                        model2.layer3[i - 1 - (2 * x)].load_state_dict(random_state_list[i + 1])
                        # print(model2.layer3[i - 1 - (2 * x)])
                indicator = ZeroNas(dataloader=data.train_loader, indicator=args.zero_proxy, num_batch=args.num_batch)
                value = indicator.get_score(model2)[args.zero_proxy]
                value_list[iii].append((p, value))

    elif args.arch in ['vgg16_signal']:
        for iii in range(max_part):
            for p in combinations(block[iii], remain_layer[iii]):
                print(p)
                model2_state_dict = model2.state_dict()
                model1_state_dict = model1.state_dict()
                for uuu in p:
                    bias_key = '{}.bias'.format(vgg16_signal_layer_name[uuu])
                    weight_key = '{}.weight'.format(vgg16_signal_layer_name[uuu])
                    model2_state_dict[weight_key] = model1_state_dict[weight_key]
                    model2_state_dict[bias_key] = model1_state_dict[bias_key]
                model2.load_state_dict(model2_state_dict)
                indicator = ZeroNas(dataloader=data.train_loader, indicator=args.zero_proxy, num_batch=args.num_batch)
                value = indicator.get_score(model2)[args.zero_proxy]
                value_list[iii].append((p, value))

    elif args.arch in ['CNN1D']:
        for iii in range(max_part):
            for p in combinations(block[iii], remain_layer[iii]):
                print(p)
                model2_state_dict = model2.state_dict()
                model1_state_dict = model1.state_dict()
                # print("Model1 state dict keys:", model1_state_dict.keys())
                for uuu in p:
                    weight_key = '{}.weight'.format(CNN1D_layer_name[uuu])
                    # bias_key = '{}.bias'.format(CNN1D_layer_name[uuu])
                    model2_state_dict[weight_key] = model1_state_dict[weight_key]
                    # model2_state_dict[bias_key] = model1_state_dict[bias_key]
                model2.load_state_dict(model2_state_dict)
                indicator = ZeroNas(dataloader=data.train_loader, indicator=args.zero_proxy, num_batch=args.num_batch)
                value = indicator.get_score(model2)[args.zero_proxy]
                value_list[iii].append((p, value))

    if args.arch == 'SigNet50':
        model1_state_dict = model1.state_dict()
        # print("Model1 state dict keys:", model1_state_dict.keys())
        print(block)
        print(max_part)
        for iii in range(max_part):
            for p in combinations(block[iii], remain_layer[iii]):  # remain 3 pretrained layers each part, 第一个部分第一层不删，所以少保留一个
                print(p)
                for bottleneck_idx in p:
                    # if bottleneck_idx == 0:
                    #     filter_keys = [
                    #         'filter.filter', 'conv1.weight', 'bn1.weight', 'bn1.bias',
                    #         'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked'
                    #     ]
                    #     for key in filter_keys:
                    #         if key in model1_state_dict:
                    #             model2.state_dict()[key].copy_(model1_state_dict[key])
                    #             # print(f"Loaded {key} into model2")
                    #         else:
                    #             print(f"Key {key} not found in model1")
                    if bottleneck_idx in [0, 1, 2]:
                        # 定义 bottleneck 的层次结构，找到该层的所有权重
                        layer_num = 1  # 每个 layer 包含多个 bottleneck
                        block_num = bottleneck_idx
                        for part in ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3']:
                            # 生成在 model1 和 model2 中对应的键
                            key_in_model1 = f'layer{layer_num}.{block_num}.{part}.weight'
                            key_in_model2 = key_in_model1
                            # 将 model1 的权重复制到 model2 中
                            if key_in_model1 in model1_state_dict:
                                model2.state_dict()[key_in_model2].copy_(model1_state_dict[key_in_model1])
                                # print(f"Loaded {key_in_model1} into model2")
                            else:
                                print(f"Key {key_in_model1} not found in model1")
                    elif bottleneck_idx in [3, 4, 5, 6]:
                        # 定义 bottleneck 的层次结构，找到该层的所有权重
                        layer_num = 2  # 每个 layer 包含多个 bottleneck
                        block_num = bottleneck_idx - 3
                        for part in ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3']:
                            # 生成在 model1 和 model2 中对应的键
                            key_in_model1 = f'layer{layer_num}.{block_num}.{part}.weight'
                            key_in_model2 = key_in_model1
                            # 将 model1 的权重复制到 model2 中
                            if key_in_model1 in model1_state_dict:
                                model2.state_dict()[key_in_model2].copy_(model1_state_dict[key_in_model1])
                                # print(f"Loaded {key_in_model1} into model2")
                            else:
                                print(f"Key {key_in_model1} not found in model1")
                    elif bottleneck_idx in [7, 8, 9, 10, 11, 12]:
                        # 定义 bottleneck 的层次结构，找到该层的所有权重
                        layer_num = 3  # 每个 layer 包含多个 bottleneck
                        block_num = bottleneck_idx - 7
                        for part in ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3']:
                            # 生成在 model1 和 model2 中对应的键
                            key_in_model1 = f'layer{layer_num}.{block_num}.{part}.weight'
                            key_in_model2 = key_in_model1
                            # 将 model1 的权重复制到 model2 中
                            if key_in_model1 in model1_state_dict:
                                model2.state_dict()[key_in_model2].copy_(model1_state_dict[key_in_model1])
                                # print(f"Loaded {key_in_model1} into model2")
                            else:
                                print(f"Key {key_in_model1} not found in model1")
                    elif bottleneck_idx in [13, 14, 15]:
                        # 定义 bottleneck 的层次结构，找到该层的所有权重
                        layer_num = 4  # 每个 layer 包含多个 bottleneck
                        block_num = bottleneck_idx - 13
                        for part in ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3']:
                            # 生成在 model1 和 model2 中对应的键
                            key_in_model1 = f'layer{layer_num}.{block_num}.{part}.weight'
                            key_in_model2 = key_in_model1
                            # 将 model1 的权重复制到 model2 中
                            if key_in_model1 in model1_state_dict:
                                model2.state_dict()[key_in_model2].copy_(model1_state_dict[key_in_model1])
                                # print(f"Loaded {key_in_model1} into model2")
                            else:
                                print(f"Key {key_in_model1} not found in model1")
                indicator = ZeroNas(dataloader=data.train_loader, indicator=args.zero_proxy, num_batch=args.num_batch)
                value = indicator.get_score(model2)[args.zero_proxy]
                value_list[iii].append((p, value))

    return value_list

def search_best(value_list):
    num = len(value_list)
    for i in range(num):
        best = +1000000000000000000000000000000000000000000000
        best_layer = None
        for j in value_list[i]:
            if best > j[1]:
                best = j[1]
                best_layer = j[0]

        print("Remaining layers: {}".format(best_layer))

    return
if __name__ == "__main__":
    main()
