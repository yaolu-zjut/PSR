from collections import OrderedDict
from args import args
import torch.nn.init as init


def get_cresnet_layer_params(target_model):
    orginal_state_list = []
    orginal_state_list.append(target_model.conv1.state_dict())
    orginal_state_list.append(target_model.bn1.state_dict())
    for child in target_model.layer1.children():
        orginal_state_list.append(child.state_dict())
    for child in target_model.layer2.children():
        orginal_state_list.append(child.state_dict())
    for child in target_model.layer3.children():
        orginal_state_list.append(child.state_dict())
    orginal_state_list.append(target_model.fc.state_dict())
    return orginal_state_list





def get_layer_params(model):
    if args.arch == 'ResNet56_signal':
        return get_cresnet_layer_params(model)
    elif args.arch == 'ResNet110_signal':
        return get_cresnet_layer_params(model)

def load_cresnet_layer_params(orginal_state_list, pruned_model, remain_layer_list, num_of_block):
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

            # print("len(orginal_state_list):", len(orginal_state_list))
            # print("remain_layer_list[2][k]:", remain_layer_list[2][k])
            # print("2 * num_of_block + 2:", 2 * num_of_block + 2)

            child.load_state_dict(orginal_state_list[remain_layer_list[2][k] + 2 * num_of_block + 2])
        k += 1

    pruned_model.fc.load_state_dict(orginal_state_list[-1])
    return pruned_model


def load_vgg_layer_params(orginal_model, orginal_conv_list, pruned_model, pruned_conv_list):
    orginal_model_state_dict = orginal_model.state_dict()
    pruned_model_state_dict = pruned_model.state_dict()
    if pruned_conv_list == [0, 2, 4, 7]:  # radio128

        for uuu in range(9):  # 一共9层conv
            if uuu == 0:
                # 匹配的加载原模型参数
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 2 or uuu == 4 or uuu == 7:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-1])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-1])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
                # print(pruned_model_state_dict[pruned_weight_key])
                # print(pruned_model_state_dict[pruned_bias_key])
    elif pruned_conv_list == [0, 1, 2, 3, 6, 7]:  # radio512
        for uuu in range(9):  # 一共9层conv
            if uuu == 0 or uuu == 1 or uuu == 2 or uuu == 3 or uuu == 6 or uuu == 7:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]

        # orginal_conv_list = ['layer1.1.0', 'layer2.0.0', 'layer2.1.0', 'layer3.0.0', 'layer3.1.0', 'layer3.2.0','layer4.0.0', 'layer4.1.0', 'layer4.2.0']
    elif pruned_conv_list == [0, 1, 2, 3, 8]:  # radio1024
        for uuu in range(9):  # 一共9层conv
            if uuu == 0 or uuu == 1 or uuu == 2 or uuu == 3:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 8:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-2])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-2])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)

    elif pruned_conv_list == [1, 4, 8]:  # all_radio128
        for uuu in range(9):  # 一共9层conv
            if uuu == 1:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 4:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-1])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-1])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
            if uuu == 8:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-2])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-2])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
    elif pruned_conv_list == [1, 4, 7]:  # all_radio512
        for uuu in range(9):  # 一共9层conv
            if uuu == 1:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 4 or uuu == 7:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-1])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-1])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)

    elif pruned_conv_list == [1, 3, 8]:  # SR-init/128
        for uuu in range(9):  # 一共9层conv
            if uuu == 1 or uuu == 3:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 8:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-2])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-2])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
    elif pruned_conv_list == [1, 3, 7]:  # RandLayer/128
        for uuu in range(9):  # 一共9层conv
            if uuu == 1 or uuu == 3:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 7:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-1])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-1])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
    elif pruned_conv_list == [2, 4, 6]:  # SR-init/512
        for uuu in range(9):  # 一共9层conv
            if uuu == 6:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 2 or uuu == 4:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-1])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-1])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
    elif pruned_conv_list == [2, 3, 6]:  # RandLayer/512
        for uuu in range(9):  # 一共9层conv
            if uuu == 6 or uuu == 3:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 2:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-1])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-1])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
    elif pruned_conv_list == [0, 1, 4, 5, 7]:
        for uuu in range(9):  # 一共9层conv
            if uuu == 0 or uuu == 1:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 4 or uuu == 7:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-1])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-1])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
            if uuu == 5:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-2])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-2])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
    elif pruned_conv_list == [0, 2, 3, 4, 8]:
        for uuu in range(9):  # 一共9层conv
            if uuu == 0 or uuu == 3 or uuu == 4:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 2:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-1])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-1])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
            if uuu == 8:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-2])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-2])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)

    elif pruned_conv_list == [1, 5, 6]:
        for uuu in range(9):  # 一共9层conv
            if uuu == 1 or uuu == 6:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 5:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-2])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-2])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
    elif pruned_conv_list == [1, 3, 7]:
        for uuu in range(9):  # 一共9层conv
            if uuu == 1 or uuu == 3:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 7:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-1])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-1])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)
    # orginal_conv_list = ['layer1.1.0', 'layer2.0.0', 'layer2.1.0', 'layer3.0.0', 'layer3.1.0', 'layer3.2.0','layer4.0.0', 'layer4.1.0', 'layer4.2.0']
    elif pruned_conv_list == [0, 1, 3, 5, 6]:
        for uuu in range(9):  # 一共9层conv
            if uuu == 0 or uuu == 1 or uuu == 3 or uuu == 6:
                orginal_bias_key = '{}.bias'.format(orginal_conv_list[uuu])
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
                pruned_model_state_dict[orginal_bias_key] = orginal_model_state_dict[orginal_bias_key]
            if uuu == 5:
                pruned_bias_key = '{}.bias'.format(orginal_conv_list[uuu-2])
                pruned_weight_key = '{}.weight'.format(orginal_conv_list[uuu-2])
                # 不匹配初始化
                init.kaiming_normal_(pruned_model_state_dict[pruned_weight_key])
                init.constant_(pruned_model_state_dict[pruned_bias_key], 0)

    pruned_model.load_state_dict(pruned_model_state_dict)
    return pruned_model

def load_cnn_layer_params(orginal_model, orginal_conv_list, pruned_model, pruned_conv_list):
    orginal_model_state_dict = orginal_model.state_dict()
    pruned_model_state_dict = pruned_model.state_dict()
    if pruned_conv_list == [0, 4]:
        for uuu in range(30):  # 一共9层conv
            if uuu in [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]:
                # 匹配的加载原模型参数
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
    elif pruned_conv_list == [0, 2]:
        for uuu in range(30):  # 一共9层conv
            if uuu in [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]:
                # 匹配的加载原模型参数
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
    elif pruned_conv_list == [0, 1]:
        for uuu in range(30):  # 一共9层conv
            if uuu in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                # 匹配的加载原模型参数
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
    elif pruned_conv_list == [0, 3]:
        for uuu in range(30):  # 一共9层conv
            if uuu in [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]:
                # 匹配的加载原模型参数
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
    elif pruned_conv_list == [0, 5]:
        for uuu in range(30):  # 一共9层conv
            if uuu in [0, 1, 2, 3, 4, 25, 26, 27, 28, 29]:
                # 匹配的加载原模型参数
                orginal_weight_key = '{}.weight'.format(orginal_conv_list[uuu])
                pruned_model_state_dict[orginal_weight_key] = orginal_model_state_dict[orginal_weight_key]
    pruned_model.load_state_dict(pruned_model_state_dict)
    return pruned_model

def load_signet_layer_params(original_state_dict, pruned_model, remain_layer_list):
    pruned_state_dict = pruned_model.state_dict()  # 获取剪枝模型的参数字典

    # 首先确保必须加载的固定层
    for key in ['filter.filter', 'conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                'bn1.num_batches_tracked']:
        if key in original_state_dict and key in pruned_state_dict:
            pruned_state_dict[key] = original_state_dict[key]  # 加载这些参数

    # 遍历 remain_layer_list，加载相应的 bottleneck 参数
    for layer_idx, bottleneck_indices in enumerate(remain_layer_list):
        for bottleneck_idx in bottleneck_indices:
            for param_key in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                              'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var',
                              'conv3.weight', 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var']:
                key = f'layer{layer_idx + 1}.{bottleneck_idx}.{param_key}'  # 构造键名称
                if key in original_state_dict and key in pruned_state_dict:
                    pruned_state_dict[key] = original_state_dict[key]  # 加载对应层的参数

    # 将更新后的 state_dict 加载到剪枝后的模型中
    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model
