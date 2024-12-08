import datetime
import os
import torch
import tqdm
import torch.nn as nn
from args import args
from cka import unbias_CKA, linear_CKA, cosine_similarity
from utils1.get_model import get_model
from utils1.get_hook import get_inner_feature_for_ResNet_signal, get_inner_feature_for_vgg16_signal, get_inner_feature_for_CNN1D, get_inner_feature_for_SigNet50
from utils1.get_dataset import get_dataset
from utils1.utils import set_gpu, get_logger

'''
# setup up:
python similarity.py --gpu 4 --arch vgg16_signal --set radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch vgg16_signal --set radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch vgg16_signal --set radio1024 --num_classes 24 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet56_signal --set radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet56_signal --set radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet56_signal --set radio1024 --num_classes 24 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet110_signal --set radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet110_signal --set radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet110_signal --set radio1024 --num_classes 24 --batch_size 128 --pretrained --evaluate 

python similarity.py --gpu 4 --arch vgg16_signal --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch vgg16_signal --set all_radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet56_signal --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet56_signal --set all_radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet110_signal --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch ResNet110_signal --set all_radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate 

信号专用模型：
python similarity.py --gpu 4 --arch CNN1D --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch CNN1D --set all_radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch CNN1D --set radio1024 --num_classes 24 --batch_size 128 --pretrained --evaluate 
python similarity.py --gpu 4 --arch SigNet50 --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate
python similarity.py --gpu 4 --arch SigNet50 --set all_radio512 --num_classes 12 --batch_size 128 --pretrained --evaluate

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
    criterion = nn.CrossEntropyLoss().cuda()
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
            elif args.arch in ['CNN1D']:
                handle_list = get_inner_feature_for_CNN1D(model, hook)
            elif args.arch in ['SigNet50']:
                handle_list = get_inner_feature_for_SigNet50(model, hook, args.arch)

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
    print(CKA_matrix_list)
    torch.save(CKA_matrix_list, 'CKA_matrix_for_visualization_{}_{}.pth'.format(args.arch, args.set))
    # torch.save(CKA_matrix_list, 'cosine_similarity_{}_{}.pth'.format(args.arch, args.set))
def CKA_heatmap(inter_feature):
    layer_num = len(inter_feature)
    CKA_matrix = torch.zeros((layer_num, layer_num))
    for ll in range(layer_num):
        for jj in range(layer_num):
            if ll < jj:
                CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = unbias_CKA(inter_feature[ll], inter_feature[jj])
                # CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = linear_CKA(inter_feature[ll], inter_feature[jj])
                # CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = cosine_similarity(inter_feature[ll], inter_feature[jj])
    CKA_matrix_for_visualization = CKA_matrix + torch.eye(layer_num)
    return CKA_matrix_for_visualization

if __name__ == "__main__":
    main()
