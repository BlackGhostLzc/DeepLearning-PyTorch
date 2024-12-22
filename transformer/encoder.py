import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

'''
bias=False表示这个线性层没有偏置项，这是Transformer模型中的一个常见选择，
因为偏置项可能会影响多头自注意力的缩放等比例性质。
'''

from transformer.util.constants import (tgt_vocab_size,d_model,n_layers,sentences, \
                            src_vocab,src_vocab_size,tgt_vocab,d_k,d_v,d_ff,n_heads)

from transformer.util.attention import get_attn_pad_mask, MultiHeadAttention, \
    PositionalEncoding, PoswiseFeedForwardNet, make_data


# encoder layer(block)
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        # 这个网络对每个位置的输入独立地应用两个线性变换和激活函数，它不依赖于序列中的位置信息。
        self.pos_fnn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V
        # enc_inputs[batch_size, src_len, d_model]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                           enc_inputs, enc_self_attn_mask)
        # enc_outputs [batch_size, len_q, d_model]
        enc_outputs = self.pos_fnn(enc_outputs)
        return enc_outputs, attn

'''
## Encoder
第一步，中文字索引进行Embedding，转换成512维度的字向量。
第二步，在子向量上面加上位置信息。
第三步，Mask掉句子中的占位符号。
第四步，通过6层的encoder（上一层的输出作为下一层的输入）。
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer() for _ in range(n_layers)]
        )

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model]
            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
enc_outputs, enc_self_attns = Encoder()(enc_inputs)
print(enc_outputs.shape)
