import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import encoder
from util.constants import tgt_vocab_size,d_model,n_layers

from util.attention import PositionalEncoding, MultiHeadAttention, \
    PoswiseFeedForwardNet, get_attn_pad_mask, get_attn_subsequence_mask

'''
我们这里是做一个机器翻译的任务，把一种语言翻译成另一种语言，
所以我们有两个vocab，一个是src_vocab，一个是tgt_vocab
'''

class DecoderLayer(nn.Module):
    def __int__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):

        pass



class Decoder(nn.Module):
    def __int__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        enc_inputs: [batch_size, src_len]，这是encoder的原始输入
        dec_inputs: [batch_size, tgt_len]
        enc_outputs: [batch_size, src_len, d_model] 这是encoder的输出结果，当做decoder的输入
        '''
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        # Sequence Mask
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)