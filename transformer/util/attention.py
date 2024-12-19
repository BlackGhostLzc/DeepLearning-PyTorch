import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

from .constants import d_ff, d_model, d_k, d_v, n_heads, sentences, src_vocab, tgt_vocab

def make_data(sentence):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2*i /d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1: , 0::2] = np.sin(pos_table[1: , 0::2]) # 表示从索引1开始到数组的末尾,取所有偶数
        pos_table[1: , 1::2] = np.cos(pos_table[1: , 1::2])
        self.pos_table = torch.FloatTensor(pos_table)       # numpy不适合做微分运算

    def forward(self, enc_inputs):
        # enc_inputs [batch_size, src_len, d_model]
        # pos_table  [max_len, d_model]
        # 取 pos_table 中的前 src_len 行，然后加到 enc_inputs 中，这里用了广播机制
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        outputs = self.fc(inputs)
        return nn.LayerNorm(d_model)(outputs + residual)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores = scores.masked_fill_(attn_mask, 1e-9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # fc层是多头自注意力机制中的一个线性层，它负责将多头的输出合并并转换回原始的模型维度，以便继续后续的处理。
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: # [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k] 注意力得分矩阵
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        # 拼接多头注意力的结果 context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model] 便于下一个EncoderLayer进行计算
        return nn.LayerNorm(d_model)(output + residual), attn




'''
这是padding mask,另一种是sequence mask
文本通常是不定长的，所以在输入一个样本长短不一的batch到网络前，要对batch中的样本进行truncating截断/padding补齐操作，
以便能形成一个张量的形式输入网络.

自注意力机制中，这里的 seq_q 和 seq_k 是相同的，但我们写的是注意力机制，所以 seq_q 和 seq_k 可能是不同的。
'''
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q : [batch_size, src_len]
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # 扩展成多维度   [batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    '''
    :param seq: [batch_size, tgt_len]
    :return:
    '''
    pass
