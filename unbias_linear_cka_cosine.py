import datetime
import os

import numpy as np
import torch
import tqdm
from args import args
from cka import unbias_CKA, linear_CKA
from utils1.get_model import get_model
from utils1.get_hook import get_inner_feature_for_ResNet_signal, get_inner_feature_for_vgg16_signal
from utils1.get_dataset import get_dataset
from utils1.utils import set_gpu, get_logger
import matplotlib.pyplot as plt
'''
# setup up:
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 4 --pretrained --evaluate 
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 8 --pretrained --evaluate 
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 16 --pretrained --evaluate 

python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 32 --pretrained --evaluate 
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 48 --pretrained --evaluate 
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 64 --pretrained --evaluate
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 80 --pretrained --evaluate
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 96 --pretrained --evaluate
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 112 --pretrained --evaluate
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate 
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 144 --pretrained --evaluate 
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 160 --pretrained --evaluate 

python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 256 --pretrained --evaluate 
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 512 --pretrained --evaluate 
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 1024 --pretrained --evaluate 
python unbias_linear_cka_cosine.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 2048 --pretrained --evaluate 


'''

def main():
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set):
        os.makedirs('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set, exist_ok=True)
    logger = get_logger('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set + '/logger' + now + '.log')
    logger.info(args)
    model = get_model(args)
    # print(model)
    model = set_gpu(args, model)
    data = get_dataset(args)
    model.eval()
    batch_count = 0

    inter_feature = []
    CKA_matrix_list = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(data.val_loader), ascii=True, total=len(data.val_loader)):
            batch_count += 1
            inputs, labels = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)

            if args.arch in ['vgg16_signal']:
                handle_list = get_inner_feature_for_vgg16_signal(model, hook, args.arch)
            elif args.arch in ['ResNet56_signal', 'ResNet110_signal']:
                handle_list = get_inner_feature_for_ResNet_signal(model, hook, args.arch)

            output = model(inputs)
            for m in range(len(inter_feature)):
                # print('-'*50)
                # print(m)
                if len(inter_feature[m].shape) != 2:

                    inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1)

            CKA_matrix_for_visualization = CKA_heatmap(inter_feature)
            # print(CKA_matrix_for_visualization)
            CKA_matrix_list.append(CKA_matrix_for_visualization)
            inter_feature = []
            for i in range(len(handle_list)):
                handle_list[i].remove()
            if batch_count == 5:
                break
    # 取矩阵平均操作
    avg_matrix = torch.zeros((CKA_matrix_list[0].shape[0], CKA_matrix_list[0].shape[1]))
    for i in range(len(CKA_matrix_list)):
        avg_matrix += CKA_matrix_list[i]
    avg_matrix = avg_matrix / len(CKA_matrix_list)
    print(avg_matrix)
    # torch.save(avg_matrix, '/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_{}_{}_{}.pth'.format(args.arch, args.set, args.batch_size))
    torch.save(avg_matrix, '/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_{}_{}_{}.pth'.format(args.arch, args.set, args.batch_size))

    # python unbias_linear_cka_cosine.py
    # matrix_unbias_32 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_ResNet56_signal_radio128_32.pth')
    # matrix_unbias_48 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_ResNet56_signal_radio128_48.pth')
    # matrix_unbias_64 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_ResNet56_signal_radio128_64.pth')
    # matrix_unbias_80 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_ResNet56_signal_radio128_80.pth')
    # matrix_unbias_96 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_ResNet56_signal_radio128_96.pth')
    # matrix_unbias_112 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_ResNet56_signal_radio128_112.pth')
    # matrix_unbias_128 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_ResNet56_signal_radio128_128.pth')
    # matrix_unbias_144 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_ResNet56_signal_radio128_144.pth')
    # matrix_unbias_160 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/unbias_CKA_ResNet56_signal_radio128_160.pth')
    #
    # matrix_linear_32 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_ResNet56_signal_radio128_32.pth')
    # matrix_linear_48 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_ResNet56_signal_radio128_48.pth')
    # matrix_linear_64 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_ResNet56_signal_radio128_64.pth')
    # matrix_linear_80 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_ResNet56_signal_radio128_80.pth')
    # matrix_linear_96 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_ResNet56_signal_radio128_96.pth')
    # matrix_linear_112 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_ResNet56_signal_radio128_112.pth')
    # matrix_linear_128 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_ResNet56_signal_radio128_128.pth')
    # matrix_linear_144 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_ResNet56_signal_radio128_144.pth')
    # matrix_linear_160 = torch.load('/public/ly/zyt/xianyu/CKA_matrix/linear_CKA_ResNet56_signal_radio128_160.pth')
    #
    #
    # # 提取每个矩阵的第一个元素并存储到一个新的列表中
    # first_elements_unbias = []
    # first_elements_linear = []
    # first_elements_unbias.append(matrix_unbias_32[0, 1].item())
    # first_elements_unbias.append(matrix_unbias_48[0, 1].item())
    # first_elements_unbias.append(matrix_unbias_64[0, 1].item())
    # first_elements_unbias.append(matrix_unbias_80[0, 1].item())
    # first_elements_unbias.append(matrix_unbias_96[0, 1].item())
    # first_elements_unbias.append(matrix_unbias_112[0, 1].item())
    # first_elements_unbias.append(matrix_unbias_128[0, 1].item())
    # first_elements_unbias.append(matrix_unbias_144[0, 1].item())
    # first_elements_unbias.append(matrix_unbias_160[0, 1].item())
    #
    # first_elements_linear.append(matrix_linear_32[0, 1].item())
    # first_elements_linear.append(matrix_linear_48[0, 1].item())
    # first_elements_linear.append(matrix_linear_64[0, 1].item())
    # first_elements_linear.append(matrix_linear_80[0, 1].item())
    # first_elements_linear.append(matrix_linear_96[0, 1].item())
    # first_elements_linear.append(matrix_linear_112[0, 1].item())
    # first_elements_linear.append(matrix_linear_128[0, 1].item())
    # first_elements_linear.append(matrix_linear_144[0, 1].item())
    # first_elements_linear.append(matrix_linear_160[0, 1].item())
    #
    # print(first_elements_unbias)
    # print(first_elements_linear)
    #
    # batch_sizes = [32, 48, 64, 80, 96, 112, 128, 144, 160]
    # plt.figure(figsize=(10, 6))
    # plt.plot(batch_sizes, first_elements_unbias, marker='o', linestyle='-', color='blue', label='Unbiased CKA')
    # plt.plot(batch_sizes, first_elements_linear, marker='o', linestyle='-', color='red', label='Linear CKA')
    # plt.xlabel('Batch Size')
    # plt.ylabel('CKA Value')
    # plt.legend()
    # # 添加网格
    # plt.grid(True)
    # # 保存图像为 PDF 文件
    # output_path = '/public/ly/zyt/xianyu/CKA_matrix/CKA_values_plot.pdf'
    # plt.savefig(output_path, format='pdf')


def CKA_heatmap(inter_feature):
    layer_num = len(inter_feature)
    CKA_matrix = torch.zeros((layer_num, layer_num))
    for ll in range(layer_num):
        for jj in range(layer_num):
            if ll < jj:
                # CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = unbias_CKA(inter_feature[ll], inter_feature[jj])
                CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = linear_CKA(inter_feature[ll], inter_feature[jj])
    CKA_matrix_for_visualization = CKA_matrix + torch.eye(layer_num)
    return CKA_matrix_for_visualization

if __name__ == "__main__":
    main()
