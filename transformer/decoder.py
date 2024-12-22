import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


from transformer.util.constants import tgt_vocab_size,d_model,n_layers

from transformer.util.attention import PositionalEncoding, MultiHeadAttention, \
    PoswiseFeedForwardNet, get_attn_pad_mask, get_attn_subsequence_mask

'''
我们这里是做一个机器翻译的任务，把一种语言翻译成另一种语言，
所以我们有两个vocab，一个是src_vocab，一个是tgt_vocab

step1. 首先将decoder的输入做一次mask self attention
step2. 把前面的encoder的输出结果做K和V，把decoder的中间mask self attention输出做Q，进行一个注意力机制计算
step3. 重复N层

为什么要这么做？
Encoder 的输出 𝐶 是对输入序列的全局上下文编码，包含了输入序列的语义信息。
'''

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):
        # enc_outputs: [batch_size, src_len, d_model]
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len]

        # step1,decoder的自注意力机制
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, dec_self_attn_mask)
        # step2,decoder-encoder交叉注意力
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, dec_self_attn_mask)

        # step3,前馈神经网络
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn




class Decoder(nn.Module):
    def __init__(self):
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

        '''
        下面这个mask是属于transformer架构中的Masked Multi-Head Attention的掩码矩阵
        属于是padding mask + sequence mask,
        为了时预测不要看到未来的信息。
        '''
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)

        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # Sequence Mask 矩阵
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        # torch.gt() 比较Tensor1和Tensor2的每一个元素,并返回一个0-1值.若Tensor1中的元素大于Tensor2中的元素,则结果取1,否则取0
        # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)

        '''
        下面这个mask主要用于encoder-decoder attention层
        get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，
        要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        dec_inputs只是提供expand的size的
        '''
        # [batch_size, tgt_len, src_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = \
                layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns

