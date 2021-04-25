import json
import os
import torch.nn.init
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
from collections import defaultdict

def varible(tensor, gpu):
    if gpu >= 0:
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def save_checkpoint(state, track_list, filename):
    with open(filename + '.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename + '.model')


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def generate_dir(work_dir):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

def display_tensorshape(is_display=True):
    # 只显示 tensor shape
    if is_display:
        old_repr = torch.Tensor.__repr__

        def tensor_info(tensor):
            return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(
                tensor)

        torch.Tensor.__repr__ = tensor_info

def self_cosine_distance(a, b):
    return torch.cosine_similarity(a, b)
def self_euclid_distance(a, b):
    return torch.dist(a, b)
def user_distance_matrix(knowledge_matrix, params):
    """
    用于计算用户和用户之间的知识水平的欧式距离
    :param knowledge_matrix: 从模型提取的记忆力矩阵
    :param params:
    :return: 一个key为id，value为子字典，子字典key为id，value为距离
    """
    user_distance = {}
    for id_x, knowledge_x in knowledge_matrix.items():
        user_distance_y = {}
        for id_y, knowledge_y in knowledge_matrix.items():
            # 把自己和自己的距离记做正无穷
            distance = varible(torch.tensor(float('inf')), params.gpu) if id_x == id_y else torch.dist(knowledge_x, knowledge_y)
            user_distance_y[id_y] = distance.cpu().detach().numpy() # 把距离放到内存上，去除梯度，转为np
            #user_distance[id_x].append({id_y:distance})
        user_distance[id_x] = user_distance_y
    return user_distance

def user_topk(user_dic, K):
    """
    使用用户之间知识水平的欧氏距离，生成用户最接近的K个用户
    :param user_dic: 知识水平距离字典
    :param K: 最相近的K个用户
    :return: 返回该用户最接近的K个用户id
    """
    user_recom_dic = {}
    for user_id_x, user_distance_row in user_dic.items():
        user_distance_row = pd.Series(user_distance_row).sort_values()
        user_recom_dic[user_id_x] = user_distance_row.keys()[:K]
    return user_recom_dic

def

def partition_arg_topk(array, K, axis=0):
    a_part = np.argpartition(array, -K, axis=axis)[-K: len(array)]
    if axis == 0:
        row_index = np.arange(array.shape[1 - axis])
        a_sec_argsort_K = np.argsort(array[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index][::-1]
    else:
        column_index = np.arange(array.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(array[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K][::-1]