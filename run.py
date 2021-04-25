import math

import numpy as np
import torch
from sklearn import metrics
from torch import nn

import utils as utils



def train(epoch_num, model, params, optimizer, q_data, qa_data):
    N = int(math.floor(len(q_data) / params.batch_size))

    # shuffle_index = np.random.permutation(q_data.shape[0])
    # q_data_shuffled = q_data[shuffle_index]
    # qa_data_shuffled = qa_data[shuffle_index]

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):
        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)
        input_q = utils.varible(torch.LongTensor(q_one_seq), params.gpu)
        input_qa = utils.varible(torch.LongTensor(qa_batch_seq), params.gpu)
        target = utils.varible(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        model.zero_grad()
        loss, filtered_pred, filtered_target, _ = model.forward(input_q, input_qa, target_1d)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params.maxgradnorm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        # print(right_pred)
        # print(right_target)
        # right_index = np.flatnonzero(right_target != -1.).tolist()
        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    # if (epoch_num + 1) % params.decay_epoch == 0:
    #     utils.adjust_learning_rate(optimizer, params.init_lr * params.lr_decay)
    # print('lr: ', params.init_lr / (1 + 0.75))
    # utils.adjust_learning_rate(optimizer, params.init_lr / (1 + 0.75))
    # print("all_target", all_target)
    # print("all_pred", all_pred)
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc


def test(model, params, optimizer, q_data, qa_data):
    N = int(math.floor(len(q_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):
        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = utils.varible(torch.LongTensor(q_one_seq), params.gpu)  # shape 32,200
        input_qa = utils.varible(torch.LongTensor(qa_batch_seq), params.gpu)  # shape 32,200
        target = utils.varible(torch.FloatTensor(target), params.gpu)  # shape 32,200

        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        loss, filtered_pred, filtered_target, memory_value = model.forward(input_q, input_qa, target_1d)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    # print("all_target", all_target)
    # print("all_pred", all_pred)
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc

def knowledge_matrix(model, params, id, q_data, qa_data):
    parallel = len(id)
    # 一次性加载
    knowledge_dict = {}
    N = int(math.floor(len(id) / parallel)) # inference 一条一条加载 一次性全加载解决id和matrix匹配问题
    model.eval()

    for idx in range(N):
        q_one_seq = q_data[idx * parallel : (idx + 1) * parallel, :]
        qa_one_seq = qa_data[idx * parallel : (idx + 1) * parallel, :]
        target = qa_data[idx * parallel: (idx + 1) * parallel, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = utils.varible(torch.LongTensor(q_one_seq), params.gpu)  # shape 1,200
        input_qa = utils.varible(torch.LongTensor(qa_one_seq), params.gpu)  # shape 1,200
        target = utils.varible(torch.FloatTensor(target), params.gpu)  # shape 1,200

        target_to_1d = torch.chunk(target, parallel, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(parallel)], 1)
        target_1d = target_1d.permute(1, 0)
        _, _, _, memory = model.forward(input_q, input_qa, target_1d)
    memory_list = torch.chunk(memory, parallel, 0)
    # knowledge_dict = {single_id:  for single_id, memory_matrix in zip(id, memory_list)}
    for single_id, memory_matrix in zip(id, memory_list):
        if single_id in knowledge_dict.keys():
            knowledge_dict[single_id] += memory_matrix.squeeze()
        else:
            knowledge_dict.update({single_id: memory_matrix.squeeze()})
    return knowledge_dict