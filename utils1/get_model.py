import torch
from model_signal.vgg16 import vgg16_signal, vgg16_signal_KD
from model_signal.resnet import ResNet_signal, ResNet56_signal_KD, ResNet110_signal_KD
from model_signal.CNN1D import ResNet1D, ResNet1D_KD
from model_signal.signet import ResNet50, ResNet50_KD
from utils1.get_params import get_cresnet_layer_params, load_cresnet_layer_params, load_vgg_layer_params, load_cnn_layer_params, load_signet_layer_params
def get_model(args):
    # Note that you can train your own models using train.py
    print(f"=> Getting {args.arch}")
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
                  ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_ResNet56_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
             elif args.set == 'all_radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_ResNet56_signal_best_lr=0.001_lecun.pth', map_location='cuda:%d' % args.gpu)
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
                  ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_ResNet110_signal_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
             elif args.set == 'all_radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
             elif args.set == 'all_radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)

    elif args.arch == 'CNN1D':
        model = ResNet1D()
        if args.pretrained:
             if args.set == 'all_radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_CNN1D_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
             elif args.set == 'all_radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_CNN1D_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
             elif args.set == 'radio1024':
                  ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/radio1024_CNN1D_best_lr=0.001.pth',map_location='cuda:%d' % args.gpu)
    elif args.arch == 'SigNet50':
        model = ResNet50()
        if args.pretrained:
             if args.set == 'all_radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_SigNet50_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
             elif args.set == 'all_radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_SigNet50_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)




    elif args.arch == 'ResNet56_signal_KD':
        model = ResNet56_signal_KD(dataset=args.set)
        if args.finetune:
            orginal_model = ResNet_signal(depth=56)
            if args.set == 'radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio128_ResNet56_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[3], [2], [8]]  # 修改
                # 3，每层保留Remaining layers: (4,)(12,)(27,)
            elif args.set == 'radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio512_ResNet56_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[0], [8], [8]]  # 修改
                # 4，每层保留Remaining layers: (0,)(1, 18)(27,)
            elif args.set == 'radio1024':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_ResNet56_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[0], [7], [0]]  # 修改
                # SR-init:
                #[9, 19, 0,]/////[[0], [0], [1]]
                # RandLayer: [[4], [2], [5]]
                # Linear_Classifier_Probes:
                # (0, 16, 18)
                # 3，每层保留Remaining layers: (1,)(18,)(27,)
                # 7，[[0, 1, 2], [8], [0, 8]]
                # 14，[[0, 1, 2, 3, 4, 5], [8], [0, 1, 2, 3, 4, 8]]
                # 21，[[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 8], [0, 1, 2, 3, 4, 8]]
            elif args.set == 'all_radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_ResNet56_signal_best_lr=0.001_lecun.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[4], [6], [2, 8]]  # 修改
                # SR-init:
                # [0, 9, 19, 18,]/////[[0], [0], [0, 1]]
                # RandLayer: [[6], [1], [2, 8]]
                # Linear_Classifier_Probes:
                # (0, 3, 9, 18)
                # k=3
                # 4，[[4], [0, 2], [8]]
                # 4，x:Remaining layers:(5,)(10, 21)(27,)[1, 1, 2]
                # 4，ramdom:Remaining layers:
                # 4，lucun:Remaining layers:
                # 7，[[2, 4, 6], [5, 8], [3, 8]]
                # 14，[[1, 2, 3, 5, 6, 8], [2, 7], [3, 4, 6, 7, 8]]
                # 21，[[0, 2, 3, 6, 7, 8], [0, 2, 3, 4, 5, 6, 7], [0, 1, 2, 5, 6, 7, 8]]
                # k=4
                # 4，Remaining layers: Remaining layers: (1,)(17,)(20,)(27,)[1, 1, 2]
                # k=5
                # 7，[[0], [0, 1, 7], [1, 8]]
                # 14，[[1, 4, 6], [2, 4, 7, 8], [0, 1, 3, 4, 7, 8]]
                # 21，[[0, 1, 3, 4, 5, 7, 8], [1, 2, 4, 5, 7, 8], [0, 1, 2, 3, 5, 6, 8]]
            elif args.set == 'all_radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_ResNet56_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[0], [0], [0]]  # 修改
                # SR-init:
                # [0, 9, 19]/////[[0], [0], [1]]
                # RandLayer: [[4], [7], [3]]
                # Linear_Classifier_Probes:
                # (0, 9, 18)
                # k=3
                # 3，[[0], [8], [8]]
                # 7，[[0, 1, 2], [8], [0, 8]]
                # 14，[[0, 1, 2, 3, 4, 5], [8], [0, 1, 2, 3, 4, 8]]
                # 21，[[0, 1, 3, 4, 6, 7, 8], [0, 1, 2, 4, 5, 6, 7], [1, 2, 4, 5, 6, 8]]
                # k=5
                # 7，[[0, 1, 8], [0, 7], [0, 8]]
                # 14，[[0, 1, 2, 8], [0, 1, 7, 8], [0, 1, 2, 3, 8]]
                # 21，[[0, 1, 2, 3, 4, 5, 8], [0, 1, 2, 3, 4, 7, 8], [0, 1, 2, 3, 4, 8]]
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
    elif args.arch == 'ResNet110_signal_KD':
        model = ResNet110_signal_KD(dataset=args.set)
        if args.finetune:
            orginal_model = ResNet_signal(depth=110)
            if args.set == 'radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio128_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[1, 2], [1], [5, 7, 17]]  # 修改
                # 7:Remaining layers: (0,)(2,)(3,)(20, 42)(44,)(54,)
            elif args.set == 'radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio512_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[0], [17], [17]]  # 修改
                # 4:Remaining layers: (0,)(1, 36)(54,)
            elif args.set == 'radio1024':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[0], [8], [0]]   # 修改
                # SR-init:
                # [18, 36, 37, 0,]/////[[0], [0], [0, 1]]
                # RandLayer: [[12], [2], [6, 8]]
                # Linear_Classifier_Probes:
                # (0, 26, 36)
                # 42:[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17], [0, 1, 2, 3, 4, 5, 16, 17], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]]
                # 27:[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17], [0, 1, 2, 3, 4, 5, 16, 17], [0, 1, 17]]
                # 14:[[0, 1, 4, 5, 17], [0, 3, 4, 16, 17], [0, 1, 17]]
                # k=3,4:[[0], [17], [17]]
                # 4:[[3], [17], [17]]
                # 5:[[0, 5], [3, 17], [17]]
                # 6:[[0, 5], [3, 16, 17], [17]]
                # 7:[[0, 4, 17], [3, 16, 17], [17]]
            elif args.set == 'all_radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[0], [1], [0, 1]]   # 修改
                # SR-init:
                # [18, 0, 36, 37,]//////[[0], [0], [0, 1]]
                # RandLayer: [[1], [2], [3, 15]]
                # Linear_Classifier_Probes:
                # (0, 19, 36, 37)

                # 4:[[7], [1], [7, 15]]
                # k=4:5:[[0, 1], [14, 17], [17]]
                # k=5:5:[[0, 1], [1], [2, 17]]
                # 6:[[0, 1, 4], [17], [3, 17]]
                # 7:[[0, 1, 3], [1, 17], [3, 17]]
                # 14:[[0, 1, 2, 3, 4], [1, 2, 3, 17], [0, 3, 4, 17]]
                # 27:[[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16], [1, 2, 3, 10, 17], [1, 2, 3, 4, 12, 15]]
                # 42:[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17]]
            elif args.set == 'all_radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_ResNet110_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[0], [0, 1], [0]]  # 修改
                # SR-init:
                # [0, 18, 37, 36,]/////[[0], [0], [0, 1]]
                # RandLayer: [[11], [13], [16, 17]]
                # Linear_Classifier_Probes:
                # (0, 18, 19, 36)

                # 4:[[0], [17], [1, 17]]
                # 4:[[1], [17], [1, 17]]
                # 5:[[0], [2, 17], [1, 17]]
                # 6:[[17], [2, 17], [1, 17]]
                # 7:[[17], [2, 17], [1, 8, 17]]
                # 14:[[0, 17], [0, 1, 2, 3, 17], [0, 1, 2, 8, 9, 17]]
                # 27:[[0, 1, 17], [0, 1, 2, 3, 4, 17], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]]
                # 42:[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17]]
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=18)
            print('Load pretrained weights from the original model')
    elif args.arch == 'vgg16_signal_KD':
        model = vgg16_signal_KD(dataset=args.set)
        if args.finetune:
            orginal_model = vgg16_signal(dataset=args.set)
            if args.set == 'radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio128_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                pruned_conv_list = [0, 2, 4, 7]
                # 4，需要调整Remaining layers: (0, 2)(4,)(7,)
            elif args.set == 'radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio512_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                pruned_conv_list = [0, 1, 2, 3, 6, 7]
                # 5，需要调整Remaining layers: (0,)(1, 2, 3, 6)(7,)
            elif args.set == 'radio1024':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/radio1024_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                pruned_conv_list = [0, 1, 3, 5, 6]
                # 5，需要调整Remaining layers: (0,)(2, 3, 4)(8,)
                # SR-init:(0,)(1, 4, 5)(7,)
                # RandLayer:(0,)(1, 2, 3)(8,)
                # LCP:(0, 1,)(3, 5,)(6,)
            elif args.set == 'all_radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                pruned_conv_list = [1, 5, 6]
                # 3，需要调整Remaining layers: (1,)(4,)(8,)
                # SR-init:(1,)(3,)(8,)
                # RandLayer:(1,)(3,)(7,)
                # LCP:(1,)(5,)(6,)
            elif args.set == 'all_radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_vgg16_signal_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                pruned_conv_list = [1, 3, 7]
                # 3，需要调整Remaining layers: (1,)(4,)(7,)
                # SR-init:(2,)(4,)(6,)
                # RandLayer:(2,)(3,)(6,)
                # LCP:(1,)(3,)(7,)
            orginal_conv_list = ['layer1.1.0', 'layer2.0.0', 'layer2.1.0', 'layer3.0.0', 'layer3.1.0', 'layer3.2.0','layer4.0.0', 'layer4.1.0', 'layer4.2.0']
            orginal_model.load_state_dict(ckpt)
            model = load_vgg_layer_params(orginal_model, orginal_conv_list, model, pruned_conv_list)
            print('Load pretrained weights from the original model')


    elif args.arch == 'CNN1D_KD':
        model = ResNet1D_KD(dataset=args.set)
        if args.finetune:
            orginal_model = ResNet1D(dataset=args.set)
            if args.set == 'all_radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_CNN1D_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                pruned_conv_list = [0, 5]
                # 需要调整Remaining layers: (0,) (4,)
                # SR-init:(0,) (3,)
                # RandLayer:(0,) (1,)
                # LCP:(0,) (5,)
            elif args.set == 'all_radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_CNN1D_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                pruned_conv_list = [0, 5]
                # 3，需要调整Remaining layers: (0,) (4,)
                # SR-init:(0,) (3,)
                # RandLayer:(0,) (1,)
                # LCP:(0,) (5,)
            elif args.set == 'radio1024':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/radio1024_CNN1D_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                pruned_conv_list = [0, 5]
                # 5，需要调整Remaining layers: (0,) (2,)
                # SR-init:(0,) (3,)
                # RandLayer:(0,) (1,)
                # LCP:(0,) (5,)
            orginal_conv_list = ['conv1.conv1', 'conv1.conv2', 'conv1.conv3', 'conv1.conv4', 'conv1.conv5', 'conv2.conv1', 'conv2.conv2', 'conv2.conv3', 'conv2.conv4', 'conv2.conv5',
                    'conv3.conv1', 'conv3.conv2', 'conv3.conv3', 'conv3.conv4', 'conv3.conv5', 'conv4.conv1', 'conv4.conv2', 'conv4.conv3', 'conv4.conv4', 'conv4.conv5',
                    'conv5.conv1', 'conv5.conv2', 'conv5.conv3', 'conv5.conv4', 'conv5.conv5', 'conv6.conv1', 'conv6.conv2', 'conv6.conv3', 'conv6.conv4', 'conv6.conv5']
            orginal_model.load_state_dict(ckpt)
            model = load_cnn_layer_params(orginal_model, orginal_conv_list, model, pruned_conv_list)
            print('Load pretrained weights from the original model')
    elif args.arch == 'SigNet50_KD':
        model = ResNet50_KD(dataset=args.set)
        if args.finetune:
            orginal_model = ResNet50(dataset=args.set)
            if args.set == 'all_radio128':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio128_SigNet50_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[2], [2], [3, 4], [1]]  # 修改
                # Remaining layers: (2,)(3, 4) (13,) (14,)[[1, 2], [0], [5], [0]]
                # SR-init:3, 6, 11, 12, 15,[[0], [0], [0, 1], [0]]
                # RandLayer:2, 4, 5, 9, 15[[1], [0, 1], [1], [1]]
                # LCP:1, 5, 8, 12, 16[[0], [1], [0, 4], [2]]
            elif args.set == 'all_radio512':
                ckpt = torch.load('/public/ly/zyt/xianyu/pretrained_signal/result/all_radio512_SigNet50_best_lr=0.001.pth', map_location='cuda:%d' % args.gpu)
                remain_list = [[1], [0], [0], [2]]  # 修改
                # Remaining layers:(0,)(4,)(8,)(15,)[[0], [0], [0], [1]]
                # SR-init:0, 6, 9, 16[[0], [2], [1], [2]]
                # RandLayer:3, 5, 11, 16[[2], [1], [3], [2]]
                # LCP:2, 4, 8, 16[[1], [0], [0], [2]]
            orginal_model.load_state_dict(ckpt)
            orginal_state_dict = orginal_model.state_dict()
            model = load_signet_layer_params(orginal_state_dict, model, remain_list)
            print('Load pretrained weights from the original model')

    else:
        assert "the model has not prepared"
    # if the model is loaded from torchvision, then the codes below do not need.
    if args.set in ['radio128', 'radio512', 'radio1024', 'all_radio128', 'all_radio512']:
        if args.pretrained:
            model.load_state_dict(ckpt)
        else:
            print('No pretrained model')
    else:
        print('Not mentioned dataset')
    return model


