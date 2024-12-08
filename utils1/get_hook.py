from args import args
import torch
cfgs = {
    'vgg16_signal': ['layer1.1.0','layer2.0.0','layer2.1.0','layer3.0.0','layer3.1.0','layer3.2.0','layer4.0.0','layer4.1.0','layer4.2.0'],
    'ResNet56_signal': [9, 9, 9],
    'ResNet110_signal': [18, 18, 18],
    'SigNet50': [3, 4, 6, 3],
}

def get_inner_feature_for_vgg16_signal(model, hook, arch):
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]

    handle_list = []
    print('cfg:', cfg)
    count = 0
    for idx, m in enumerate(model.named_modules()):
        name, module = m[0], m[1]
        if count < len(cfg):
            if name == cfg[count]:
                print(module)
                handle = module.register_forward_hook(hook)
                handle_list.append(handle)
                count += 1
        else:
            break
    return handle_list
def get_inner_feature_for_ResNet_signal(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]
    print('cfg:', cfg)
    handle = model.conv1.register_forward_hook(hook)
    handle_list.append(handle)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)
        handle_list.append(handle)
    return handle_list
def get_inner_feature_for_CNN1D(model, hook):
    handle_list = []
    # 直接对整个 conv1 模块添加钩子
    handle = model.conv1.register_forward_hook(hook)
    handle_list.append(handle)
    # 直接对整个 conv2 模块添加钩子
    handle = model.conv2.register_forward_hook(hook)
    handle_list.append(handle)
    # 直接对整个 conv3 模块添加钩子
    handle = model.conv3.register_forward_hook(hook)
    handle_list.append(handle)
    # 直接对整个 conv4 模块添加钩子
    handle = model.conv4.register_forward_hook(hook)
    handle_list.append(handle)
    # 直接对整个 conv5 模块添加钩子
    handle = model.conv5.register_forward_hook(hook)
    handle_list.append(handle)
    # 直接对整个 conv6 模块添加钩子
    handle = model.conv6.register_forward_hook(hook)
    handle_list.append(handle)

    return handle_list


def get_inner_feature_for_SigNet50(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]
    print('cfg:', cfg)
    # handle = model.conv1.register_forward_hook(hook)  # here!!!
    # handle_list.append(handle)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[3]):
        handle = model.layer4[i].register_forward_hook(hook)
        handle_list.append(handle)
    return handle_list


def get_inner_feature_for_resnet(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]
    print('cfg:', cfg)
    handle = model.conv1.register_forward_hook(hook)  # here!!!
    handle_list.append(handle)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[3]):
        handle = model.layer4[i].register_forward_hook(hook)
        handle_list.append(handle)
    return handle_list




def get_inner_feature_for_vgg(model, hook, arch):
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]

    handle_list = []
    print('cfg:', cfg)
    count = 0
    for idx, m in enumerate(model.named_modules()):
        name, module = m[0], m[1]
        if count < len(cfg):
            if name == cfg[count]:
                print(module)
                handle = module.register_forward_hook(hook)
                handle_list.append(handle)
                count += 1
        else:
            break
    return handle_list



if __name__ == "__main__":
    # demo
    import torch
    from torchvision.models import *

    input = torch.randn((2, 3, 224, 224))
    inter_feature = []
    model = vgg11_bn()

    def hook(module, input, output):
        inter_feature.append(output.clone().detach())
    get_inner_feature_for_vgg(model, hook, 'cvgg19')
    model(input)
