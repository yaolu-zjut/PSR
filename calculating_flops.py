import argparse
import torch
from thop import profile, clever_format
from utils1.get_dataset import get_dataset
from trainer.trainer import validate_signal_data
from utils1.utils import set_gpu
from model_signal.vgg16 import vgg16_signal, vgg16_signal_KD
from model_signal.resnet import ResNet_signal, ResNet56_signal_KD, ResNet110_signal_KD
from model_signal.CNN1D import ResNet1D, ResNet1D_KD
from model_signal.signet import ResNet50, ResNet50_KD

'''
python calculating_flops.py --gpu 0 --arch vgg16_signal --set radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 0 --arch vgg16_signal --set radio512 --input_signal_size 512  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch vgg16_signal --set radio1024 --input_signal_size 1024 --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch vgg16_signal --set all_radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch vgg16_signal --set all_radio512 --input_signal_size 512  --pretrained --evaluate
python calculating_flops.py --gpu 2 --arch vgg16_signal_KD --set radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 2 --arch vgg16_signal_KD --set radio512 --input_signal_size 512  --pretrained --evaluate
python calculating_flops.py --gpu 2 --arch vgg16_signal_KD --set radio1024 --input_signal_size 1024 --pretrained --evaluate
python calculating_flops.py --gpu 2 --arch vgg16_signal_KD --set all_radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 2 --arch vgg16_signal_KD --set all_radio512 --input_signal_size 512  --pretrained --evaluate

python calculating_flops.py --gpu 4 --arch ResNet56_signal --set radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch ResNet56_signal --set radio512 --input_signal_size 512  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch ResNet56_signal --set radio1024 --input_signal_size 1024 --pretrained --evaluate

python calculating_flops.py --gpu 3 --arch ResNet56_signal_KD --set all_radio128 --input_signal_size 128  --pretrained --evaluate

python calculating_flops.py --gpu 2 --arch ResNet56_signal_KD --set radio512 --input_signal_size 512 --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch ResNet110_signal --set radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch ResNet110_signal --set radio512 --input_signal_size 512 --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch ResNet110_signal --set radio1024 --input_signal_size 1024 --pretrained --evaluate
python calculating_flops.py --gpu 2 --arch ResNet110_signal_KD --set radio128 --input_signal_size 128 --pretrained --evaluate
python calculating_flops.py --gpu 2 --arch ResNet110_signal_KD --set radio512 --input_signal_size 512 --pretrained --evaluate

python calculating_flops.py --gpu 4 --arch ResNet56_signal --set all_radio128 --input_signal_size 128  --pretrained --evaluate

python calculating_flops.py --gpu 4 --arch ResNet56_signal --set all_radio512 --input_signal_size 512  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch ResNet110_signal --set all_radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch ResNet110_signal --set all_radio512 --input_signal_size 512  --pretrained --evaluate

信号专用数据集：
python calculating_flops.py --gpu 4 --arch CNN1D --set all_radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch CNN1D --set all_radio512 --input_signal_size 512  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch CNN1D --set radio1024 --input_signal_size 1024  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch CNN1D_KD --set all_radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch CNN1D_KD --set all_radio512 --input_signal_size 512  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch CNN1D_KD --set radio1024 --input_signal_size 1024  --pretrained --evaluate

python calculating_flops.py --gpu 4 --arch SigNet50 --set all_radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch SigNet50 --set all_radio512 --input_signal_size 512  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch SigNet50_KD --set all_radio128 --input_signal_size 128  --pretrained --evaluate
python calculating_flops.py --gpu 4 --arch SigNet50_KD --set all_radio512 --input_signal_size 512  --pretrained --evaluate

'''

parser = argparse.ArgumentParser(description='Calculating flops and params')
parser.add_argument('--input_signal_size',type=int,default=32,help='The input_signal_size')
parser.add_argument("--gpu", default=None, type=int, help="Which GPU to use for training")
parser.add_argument("--arch", default=None, type=str, help="arch")
parser.add_argument("--pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--num_classes", default=10, type=int, help="number of class")
parser.add_argument("--finetune", action="store_true", help="finetune pre-trained model")
parser.add_argument("--set", help="name of dataset", type=str, default='cifar10')
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 10)")
args = parser.parse_args()
torch.cuda.set_device(args.gpu)

if args.arch == 'vgg16_signal':
    model = vgg16_signal()
    if args.pretrained:
        if args.set == 'radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio128_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio512_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio1024':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
elif args.arch == 'ResNet56_signal':
    model = ResNet_signal(depth=56)
    if args.pretrained:
        if args.set == 'radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio128_ResNet56_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio512_ResNet56_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio1024':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_ResNet56_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_ResNet56_signal_best_lr=0.001_random.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_ResNet56_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
elif args.arch == 'ResNet110_signal':
    model = ResNet_signal(depth=110)
    if args.pretrained:
        if args.set == 'radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio128_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio512_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio1024':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)

elif args.arch == 'vgg16_signal_KD':
    model = vgg16_signal_KD()
    if args.pretrained:
        if args.set == 'radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_model/vgg16_signal_KD/radio128/K3_vgg16_signal_KD_radio128_67%.pt', map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_model/vgg16_signal_KD/radio512/K3_vgg16_signal_KD_radio512_67%.pt', map_location='cuda:%d' % args.gpu)
        elif args.set == 'radio1024':
            ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/vgg16_signal_KD/Linear_Classifier_Probes_vgg16_signal_KD_radio1024_44%.pt', map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/vgg16_signal_KD/Linear_Classifier_Probes_vgg16_signal_KD_all_radio128_67%.pt', map_location='cuda:%d' % args.gpu)
        elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/vgg16_signal_KD/Linear_Classifier_Probes_vgg16_signal_KD_all_radio512_67%.pt', map_location='cuda:%d' % args.gpu)

elif args.arch == 'ResNet56_signal_KD':
    model = ResNet56_signal_KD(dataset=args.set)
    if args.set == 'radio128':
        ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_model/ResNet56_signal_KD/radio128/cosine_ResNet56_signal_KD_radio128_89%.pt', map_location='cuda:%d' % args.gpu)
    elif args.set == 'radio512':
        ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_model/ResNet56_signal_KD/radio512/K3_ResNet56_signal_KD_radio512_86%.pt', map_location='cuda:%d' % args.gpu)
    elif args.set == 'radio1024': 
        ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/ResNet56_signal_KD/Linear_Classifier_Probes_ResNet56_signal_KD_radio1024_93%.pt', map_location='cuda:%d' % args.gpu)
    elif args.set == 'all_radio128':
        ckpt = torch.load("/public/ly/zyt/xianyu/pretrained_model/ResNet56_signal_KD/all_radio128/K4_ResNet56_signal_KD_all_radio128_86%.pt", map_location='cuda:%d' % args.gpu)
    elif args.set == 'all_radio512':
        ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/ResNet56_signal_KD/Linear_Classifier_Probes_ResNet56_signal_KD_all_radio512_89%.pt', map_location='cuda:%d' % args.gpu)
elif args.arch == 'ResNet110_signal_KD':
    model = ResNet110_signal_KD(dataset=args.set)
    if args.set == 'radio128':
        ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_model/ResNet110_signal_KD/radio128/K6_ResNet110_signal_KD_radio128_87%.pt', map_location='cuda:%d' % args.gpu)
    elif args.set == 'radio512':
        ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_model/ResNet110_signal_KD/radio512/K3_ResNet110_signal_KD_radio512_93%.pt', map_location='cuda:%d' % args.gpu)
    elif args.set == 'radio1024':
        ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/ResNet110_signal_KD/Linear_Classifier_Probes_ResNet110_signal_KD_radio1024_93%.pt', map_location='cuda:%d' % args.gpu)
    elif args.set == 'all_radio128':
        ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/ResNet110_signal_KD/Linear_Classifier_Probes_ResNet110_signal_KD_all_radio128_93%.pt', map_location='cuda:%d' % args.gpu)
    elif args.set == 'all_radio512':
        ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/ResNet110_signal_KD/Linear_Classifier_Probes_ResNet110_signal_KD_all_radio512_93%.pt', map_location='cuda:%d' % args.gpu)


elif args.arch == 'CNN1D':
    model = ResNet1D()
    if args.pretrained:
         if args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_CNN1D_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
         elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_CNN1D_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
         elif args.set == 'radio1024':
              ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/radio1024_CNN1D_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
elif args.arch == 'CNN1D_KD':
    model = ResNet1D_KD()
    if args.pretrained:
         if args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/CNN1D_KD/Linear_Classifier_Probes_CNN1D_KD_all_radio128_67%.pt', map_location='cuda:%d' % args.gpu)
         elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/CNN1D_KD/Linear_Classifier_Probes_CNN1D_KD_all_radio512_67%.pt', map_location='cuda:%d' % args.gpu)
         elif args.set == 'radio1024':
              ckpt = torch.load('/public/ly/zyt/xianyu/Linear_Classifier_Probes/CNN1D_KD/Linear_Classifier_Probes_CNN1D_KD_radio1024_67%.pt',map_location='cuda:%d' % args.gpu)
elif args.arch == 'SigNet50':
    model = ResNet50()
    if args.pretrained:
         if args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_SigNet50_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
         elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_SigNet50_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
elif args.arch == 'SigNet50_KD':
    model = ResNet50_KD()
    if args.pretrained:
         if args.set == 'all_radio128':
            ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_model/SigNet50_KD/all_radio128/K4_SigNet50_KD_all_radio128_76%.pt', map_location='cuda:%d' % args.gpu)
         elif args.set == 'all_radio512':
            ckpt = torch.load('/public/ly/zyt/xianyu/SR_init/SigNet50_KD/SR_SigNet50_KD_all_radio512_75%.pt', map_location='cuda:%d' % args.gpu)

# print(model)
model = set_gpu(args, model)
model.eval()
model.load_state_dict(ckpt)
criterion = torch.nn.CrossEntropyLoss().cuda()
data = get_dataset(args)
print(model)

if args.evaluate:
    if args.set in ['radio128', 'radio512', 'radio1024', 'all_radio128', 'all_radio512']:
        correct = validate_signal_data(data.val_loader, model, criterion, data.dataset_sizes)
print('Acc is {}'.format(correct))


input_signal_size = args.input_signal_size
print('signal size is {}'.format(input_signal_size))
if args.arch in ['CNN1D_KD', 'CNN1D', 'SigNet50', 'SigNet50_KD']:
    input_signal = torch.randn(1, 2, input_signal_size).cuda()
else:
    input_signal = torch.randn(1, 1, 2, input_signal_size).cuda()

flops, params = profile(model, inputs=(input_signal,))
flops, params = clever_format([flops, params], "%.2f")
# print('Params: %.2f' % (params))
# print('Flops: %.2f' % (flops))
print('Params: {}'.format(params))
print('Flops: {}'.format(flops))
# latency = compute_latency_ms_pytorch(model, input_signal, iterations=None)
# print('Latency: %.2f' % (latency))



