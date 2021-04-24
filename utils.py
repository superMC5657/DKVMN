import json
import os
import torch.nn.init


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