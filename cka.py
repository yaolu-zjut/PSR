import matplotlib as mpl
import torch.nn.functional as F


mpl.use('Agg')
import torch

def linear_HSIC(X, Y):
    n = X.shape[0]
    L_X = torch.mm(X, X.T)
    L_Y = torch.mm(Y, Y.T)
    unit = torch.ones([n, n])
    I = torch.eye(n)
    H = (I - unit / n).cuda()
    M = torch.trace(torch.mm(torch.mm(L_X, H), torch.mm(L_Y, H)))
    return M / (n - 1) / (n - 1)


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)


def unbias_CKA(X, Y):
    hsic = unbiased_HSIC(X, Y)
    var1 = torch.sqrt(unbiased_HSIC(X, X))
    var2 = torch.sqrt(unbiased_HSIC(Y, Y))
    return hsic / (var1 * var2)


def cosine_similarity(matrix1, matrix2):
    """
    计算两个二维特征矩阵之间的余弦相似度

    参数:
    matrix1: torch.Tensor, 第一个特征矩阵 (m x n)
    matrix2: torch.Tensor, 第二个特征矩阵 (m x n)

    返回值:
    float, 两个矩阵之间的余弦相似度平均值
    """
    # 确保输入矩阵是浮点张量，并且维度正确
    matrix1 = matrix1.float()
    matrix2 = matrix2.float()

    # 对每一行进行归一化
    matrix1_norm = torch.nn.functional.normalize(matrix1, p=2, dim=1)
    matrix2_norm = torch.nn.functional.normalize(matrix2, p=2, dim=1)

    # 如果矩阵的第二维大小不同，则对较小的矩阵进行填充
    if matrix1_norm.shape[1] < matrix2_norm.shape[1]:
        # 计算需要填充的大小
        padding = matrix2_norm.shape[1] - matrix1_norm.shape[1]
        # 对matrix1_norm进行填充
        matrix1_norm = F.pad(matrix1_norm, (0, padding))
    elif matrix1_norm.shape[1] > matrix2_norm.shape[1]:
        # 计算需要填充的大小
        padding = matrix1_norm.shape[1] - matrix2_norm.shape[1]
        # 对matrix2_norm进行填充
        matrix2_norm = F.pad(matrix2_norm, (0, padding))

    # 确认填充后的形状
    # print(matrix1_norm.shape)
    # print(matrix2_norm.shape)
    # 计算归一化后的矩阵的点积
    dot_product = torch.mm(matrix1_norm, matrix2_norm.transpose(0, 1))

    # 计算余弦相似度的平均值
    cos_sim = dot_product.diag().mean().item()

    return cos_sim



def unbiased_HSIC(X, Y):
    """Unbiased estimator of Hilbert-Schmidt Independence Criterion
    Song, Le, et al. "Feature selection via dependence maximization." 2012.
    """
    kernel_XX = torch.mm(X, X.T)
    kernel_YY = torch.mm(Y, Y.T)

    tK = kernel_XX - torch.diag_embed(torch.diag(kernel_XX))
    tL = kernel_YY - torch.diag_embed(torch.diag(kernel_YY))

    N = kernel_XX.shape[0]
    # print(torch.sum(tK, 0).dot(torch.sum(tL, 1)) / torch.sum(tK @ tL))  # same

    hsic = (
        torch.trace(tK @ tL)
        + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
        - (2 * torch.sum(tK @ tL) / (N - 2))
    )

    return hsic / (N * (N - 3))