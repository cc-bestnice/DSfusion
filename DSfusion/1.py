# import torch
# print("Is CUDA available:", torch.cuda.is_available())
# import torch
# print(torch.__version__)
# print(torch.version.cuda)

#特征融合部分添加的注意力机制
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class ConvLayer1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, use_bias=True, norm=None, act_func=None):
#         super(ConvLayer1D, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=use_bias)
#         self.norm = norm
#         self.act_func = act_func

#     def forward(self, x):
#         x = self.conv(x)
#         if self.norm:
#             x = self.norm(x)
#         if self.act_func:
#             x = self.act_func(x)
#         return x

# class LiteMLA(nn.Module):
#     def __init__(self, in_channels=8, out_channels=8, heads=2, dim=4, scales=(1, 3, 5)):
#         super(LiteMLA, self).__init__()
#         self.eps = 1.0e-6  # 为了数值稳定性
#         self.heads = heads
#         self.dim = dim
#         total_dim = self.heads * self.dim
#         self.qkv = ConvLayer1D(
#             in_channels, 3 * total_dim, kernel_size=1,
#             use_bias=True, norm=None, act_func=F.relu  # 在这里添加激活函数
#         )
#         # 多尺度聚合层
#         self.aggreg = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(3 * total_dim, 3 * total_dim, kernel_size=scale, padding=scale//2, groups=3*total_dim),
#                 nn.ReLU()  # 在这里添加激活函数
#             ) for scale in scales
#         ])
#         # 输出投影
#         self.proj = ConvLayer1D(
#             total_dim * len(scales), out_channels, kernel_size=1,
#             use_bias=True, norm=None, act_func=F.relu  # 在这里添加激活函数
#         )
        
#     def forward(self, x):
#         B, T, C = x.shape
#         x = x.transpose(1, 2)  # (B, C, T)
#         qkv = self.qkv(x)  # (B, 3 * total_dim, T)
#         # 应用多尺度聚合层并计算注意力
#         attention_results = []
#         for agg in self.aggreg:
#             # 对 qkv 应用多尺度卷积
#             qkv_scale = agg(qkv)  # (B, 3 * total_dim, T)
#             # 分离 Q, K, V
#             q, k, v = torch.chunk(qkv_scale, 3, dim=1)
#             q = q.view(B, self.heads, self.dim, T)
#             k = k.view(B, self.heads, self.dim, T)
#             v = v.view(B, self.heads, self.dim, T)
#             # 先计算 K 和 V 的关系
#             k_trans = k.transpose(-2, -1)  # (B, heads, T, dim)
#             kv_product = torch.matmul(k_trans, v)  # (B, heads, dim, dim)
#             # 然后计算 Q 和 K 的结果
#             q_kv_product = torch.matmul(q, kv_product)  # (B, heads, dim, T)
#             attention_results.append(q_kv_product)

#         # 合并多尺度注意力结果
#         x = torch.cat(attention_results, dim=1)  # (B, heads * dim * len(scales), T)
        
#         x = x.reshape(B, -1, T)  #合并头部和维度信息
#         x = self.proj(x)  #投影到输出通道
#         x = x.transpose(1, 2)  #(B, T, C)将结果转置回(B, T, C)形状以输出
#         return x

# lite_mla = LiteMLA(in_channels=8, out_channels=8, heads=2, dim=4, scales=(1,))
# # 假设 x 是输入张量，形状为 [64, 2, 8]
# x = torch.randn(64, 2, 8)
# output = lite_mla(x)
# print(output.shape)  # 应该输出 torch.Size([64, 2, 8])


#添加傅里叶变换
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# def process_with_fft(x_hist, fc_layer, relu, dropout):
#     """
#     对输入数据进行傅里叶变换，实部和虚部分别经过全连接层、ReLU激活和Dropout处理，
#     然后进行残差连接，最后进行逆傅里叶变换返回时域。
#     参数:
#     - x_hist: 输入数据 (batch_size, seq_len, channels)
#     - fc_layer: 全连接层
#     - relu: ReLU 激活函数
#     - dropout: Dropout 层
    
#     返回:
#     - x_his_final: 经过处理后返回时域的最终结果
#     """
#     # 对 x_hist 进行傅里叶变换
#     x_his_fft = torch.fft.fft(x_hist, dim=-1)
#     # 分别处理实部和虚部
#     real_part = dropout(relu(fc_layer(x_his_fft.real)))
#     imag_part = dropout(relu(fc_layer(x_his_fft.imag)))
#     # 合并处理后的实部和虚部
#     x_his_fft_final = real_part + imag_part * 1j
#     # 残差连接
#     x_his_fft_residual = x_his_fft + x_his_fft_final
#     # 进行逆傅里叶变换
#     x_his_final = torch.fft.ifft(x_his_fft_residual, dim=-1).real
#     return x_his_final

# # 假设 x_hist 已经被定义并且大小为 ([64, 9, 1])
# x_hist = torch.randn(64, 9, 1)
# x_hist1 = x_hist.clone()
# fc_layer = nn.Linear(x_hist1.size(-1), x_hist1.size(-1))
# relu = nn.ReLU()
# dropout = nn.Dropout(p=0.5)
# # 调用 process_with_fft 函数
# x_his_final = process_with_fft(x_hist1, fc_layer, relu, dropout)
# print(x_his_final.shape)


#添加SRAT模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STAR Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def process_input(self, input):
        """
        对输入进行 STAR 模块的处理。
        """
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        
        return combined_mean_cat

    def forward(self, x_hist, x_hist1):
        """
        对两个输入 x_hist 和 x_hist1 进行处理，并返回拼接后的输出结果。
        """
        output_x_hist = self.process_input(x_hist)
        output_x_hist1 = self.process_input(x_hist1)

        # 在特征维度上拼接两个处理后的张量
        output = torch.cat([output_x_hist, output_x_hist1], dim=-1)

        return output

# 示例输入张量
x_hist = torch.randn(64, 9, 1)  # 假设 (batch_size, channels, d_series)
x_hist1 = torch.randn(64, 9, 1)  # 假设 (batch_size, channels, d_series)

# 实例化 STAR 模块
d_series = x_hist.size(-1)
d_core = 16  # 假设核心维度为 16
star_module = STAR(d_series=d_series, d_core=d_core)
# 前向传播
output= star_module(x_hist, x_hist1)
print(output.shape)
