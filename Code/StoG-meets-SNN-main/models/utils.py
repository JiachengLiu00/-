import torch
import torch.nn as nn

class MergeTemporalDim(nn.Module):  #用于将时间步和其他维度合并为一个维度，这在某些处理时间序列数据的任务中是有用的
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module): #将合并后的张量重塑为原来的时间维度和其他维度，通常用于恢复处理后的时间序列数据的形状。
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)