import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################
##  这个是新添加的； Splice激活函数； 该函数加到 卷积层的ELU激活函数的后面
######################################
from torch import exp
class Splice(nn.Module):
    def __init__(self):
        super(Splice, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1.0)

    def forward(self, input):
        return exp(self.weight *input) / (1 + abs(input))
#####################################
##  这个是新添加的； Splice激活函数； 该函数加到 卷积层的ELU激活函数的后面
######################################


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3, ########### kernel_size=3 5, 7,9
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.activation2 = Splice()  ##  这个是新添加的；
        self.activation3 = nn.Tanh() ##  这个是新添加的；
        self.activation4 = nn.Sigmoid() ##  这个是新添加的；
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        #self.maxPool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = x = x.permute(0, 2, 1)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x) ##  nn.ELU()
        # x = self.maxPool(x)
        x = self.activation2(x) ##  这个是新添加的； 新模型启用这个；Splinformer  Prinformer
        x = self.maxPool(y+x)  # Prinformer 把这个注释掉； MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)  这一步压缩了序列的长度
        ## x = self.norm(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x) ### 残差

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None

        ## 计算模型的参数的数量
        # 计算网络参数
        total = sum([param.nelement() for param in self.attn_layers[0].parameters()])
        # 精确地计算：1MB=1024KB=1048576字节
        print('Number of parameter: % .4fM' % (total / 1e6))
        # 计算网络参数
        total = sum([param.nelement() for param in self.conv_layers[0].parameters()])
        # 精确地计算：1MB=1024KB=1048576字节
        print('Number of parameter: % .4fM' % (total / 1e6))

        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            # for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
            #     x, attn = attn_layer(x, attn_mask=attn_mask) ## 编码层
            #     x = conv_layer(x)  ## 卷积层
            #     attns.append(attn)
            if len(self.attn_layers)==3: ### Informer 原始版本 ； 把'--e_layers'设置为3
                x, attn = self.attn_layers[0](x, attn_mask=attn_mask)
                x = self.conv_layers[0](x)
                attns.append(attn)
                x, attn = self.attn_layers[1](x, attn_mask=attn_mask)
                x = self.conv_layers[1](x)
                attns.append(attn)
            if len(self.attn_layers)==3: ### Informer, Informer减掉一个编码层 ； 把'--e_layers'设置为3
                x, attn = self.attn_layers[0](x, attn_mask=attn_mask)
                x = self.conv_layers[0](x)
                attns.append(attn)
                #x, attn = self.attn_layers[1](x, attn_mask=attn_mask) # Informer减掉一个编码层
                x = self.conv_layers[1](x)
                attns.append(attn)
            if len(self.attn_layers)==3: ### Informer, Informer减掉一个卷积层 ； 把'--e_layers'设置为3
                x, attn = self.attn_layers[0](x, attn_mask=attn_mask)
                x = self.conv_layers[0](x)
                attns.append(attn)
                x, attn = self.attn_layers[1](x, attn_mask=attn_mask)
                #x = self.conv_layers[1](x)  ## Informer减掉一个卷积层
                attns.append(attn)
            if len(self.attn_layers)==3: ### Informer, Informer 减掉一个编码层，减掉一个卷积层  ； 把'--e_layers'设置为3
                x, attn = self.attn_layers[0](x, attn_mask=attn_mask)
                x = self.conv_layers[0](x)
                attns.append(attn)
                #x, attn = self.attn_layers[1](x, attn_mask=attn_mask) # Informer 减掉一个编码层，减掉一个卷积层
                #x = self.conv_layers[1](x)
                attns.append(attn)

            self.attn_layers[0]

            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask) ### 编码层
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
