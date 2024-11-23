from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#在输入张量的时间和特征维度上应用批量归一化
class TimeBatchNorm2d(nn.BatchNorm1d):
    """A batch normalization layer that normalizes over the last two dimensions of a sequence in PyTorch, mimicking Keras behavior.

    This class extends nn.BatchNorm1d to apply batch normalization across time and feature dimensions.

    Attributes:
        num_time_steps (int): Number of time steps in the input.
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, normalized_shape: tuple[int, int]):
        """Initializes the TimeBatchNorm2d module.
        Args:
            normalized_shape (tuple[int, int]): A tuple (num_time_steps, num_channels)
                representing the shape of the time and feature dimensions to normalize.
        """
        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        """Applies the batch normalization over the last two dimensions of the input tensor.
        Args:
            x (Tensor): A 3D tensor with shape (N, S, C), where N is the batch size, S is the number of time steps, and C is the number of channels.
        Returns:
            Tensor: A 3D tensor with batch normalization applied over the last two dims.
        Raises:
            ValueError: If the input tensor is not 3D.
        """
        #参数x是一个形状为 (N, S, C) 的3D张量，其中 N 是批量大小，S 是时间步数，C 是通道数。
        if x.ndim != 3: #先检查输入张量 x 的维度是否为3，如果不是，则抛出一个值错误
            raise ValueError(f"Expected 3D input tensor, but got {x.ndim}D tensor instead.")
        # Rx 被重塑为 (N, S*C, 1)，这样做是为了将时间和特征维度合并，以便可以在这个合并的维度上应用批量归一化。
        # S 和 C 是被乘在一起的，因此批量归一化能够跨时间步和通道维度进行。
        x = x.reshape(x.shape[0], -1, 1)
        #使用 super().forward(x) 调用父类 BatchNorm1d 的 forward 方法来应用归一化
        x = super().forward(x)
        # 输出 x 被重塑回原始的 (N, S, C) 维度
        x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels)
        return x

# #一维卷积
class ConvLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, use_bias=True, norm=None, act_func=None):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=use_bias)
        self.norm = norm
        self.act_func = act_func

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act_func:
            x = self.act_func(x)
        return x

# #注意力机制
class LiteMLA(nn.Module):
    def __init__(self, in_channels=8, out_channels=8, heads=4, dim=4, scales=(1, 3, 5)):
        super(LiteMLA, self).__init__()
        self.eps = 1.0e-6  # 为了数值稳定性
        self.heads = heads
        self.dim = dim
        total_dim = self.heads * self.dim
        self.qkv = ConvLayer1D(
            in_channels, 3 * total_dim, kernel_size=1,
            use_bias=True, norm=None, act_func=F.relu  # 在这里添加激活函数
        )
        # 多尺度聚合层
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(3 * total_dim, 3 * total_dim, kernel_size=scale, padding=scale//2, groups=3*total_dim),
                nn.ReLU()  # 在这里添加激活函数
            ) for scale in scales
        ])
        # 输出投影
        self.proj = ConvLayer1D(
            total_dim * len(scales), out_channels, kernel_size=1,
            use_bias=True, norm=None, act_func=F.relu  # 在这里添加激活函数
        )
        
    def forward(self, x):
        B, T, C = x.shape
        x = x.transpose(1, 2)  # (B, C, T)
        qkv = self.qkv(x)  # (B, 3 * total_dim, T)
        # 应用多尺度聚合层并计算注意力
        attention_results = []
        for agg in self.aggreg:
            # 对 qkv 应用多尺度卷积
            qkv_scale = agg(qkv)  # (B, 3 * total_dim, T)
            # 分离 Q, K, V
            q, k, v = torch.chunk(qkv_scale, 3, dim=1)
            q = q.view(B, self.heads, self.dim, T)
            k = k.view(B, self.heads, self.dim, T)
            v = v.view(B, self.heads, self.dim, T)
            # 对 Q 和 K 应用 ReLU 激活函数
            q = torch.relu(q)
            k = torch.relu(k)
            # 先计算 K 和 V 的关系
            k_trans = k.transpose(-2, -1)  # (B, heads, T, dim)
            kv_product = torch.matmul(k_trans, v)  # (B, heads, dim, dim)
            # 然后计算 Q 和 K 的结果
            q_kv_product = torch.matmul(q, kv_product)  # (B, heads, dim, T)
            attention_results.append(q_kv_product)

        # 合并多尺度注意力结果
        x = torch.cat(attention_results, dim=1)  # (B, heads * dim * len(scales), T)
        
        x = x.reshape(B, -1, T)  #合并头部和维度信息
        x = self.proj(x)  #投影到输出通道
        x = x.transpose(1, 2)  #(B, T, C)将结果转置回(B, T, C)形状以输出
        return x

#特征融合
class FeatureMixing(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
        # norm_type: type[nn.Module] = nn.BatchNorm1d,
    ):
        super().__init__()
        self.norm_before = (
            norm_type((sequence_length, input_channels))
            if normalize_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((sequence_length, output_channels))
            if not normalize_before
            else nn.Identity()
        )
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_channels)
        self.projection = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else nn.Identity()
        )
        # Initialize the attention layer if required
        self.attention = LiteMLA(
                in_channels=8,  # Attention on the output of fc2
                out_channels=8,
                heads=2,
                dim=4,
                scales=(1,3,5)
            )
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.projection(x)
        x = self.norm_before(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = self.attention(x)  
        x = x_proj + x
        # print(x.shape)
        
        return self.norm_after(x)

#整合静态特征和动态特征，并对这些融合后的特征进行处理
class ConditionalFeatureMixing(nn.Module):
    """Conditional feature mixing module that incorporates static features.
        This module extends the feature mixing process by including static features. It uses
    a linear transformation to integrate static features into the dynamic feature space,
    then applies the feature mixing on the concatenated features.
    Args:
        input_channels: The number of input channels of the dynamic features.
        output_channels: The number of output channels after feature mixing.
        static_channels: The number of channels in the static feature input.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in feature mixing.
        dropout_rate: The dropout probability used in the feature mixing operation.
    """
    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        static_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.fr_static = nn.Linear(static_channels, output_channels)
        self.fm = FeatureMixing(
            sequence_length,
            input_channels + output_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(
        self, x: torch.Tensor, x_static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies conditional feature mixing using both dynamic and static inputs.
        Args:
            x: A tensor representing dynamic features, typically with shape
               [batch_size, time_steps, input_channels].
            x_static: A tensor representing static features, typically with shape
               [batch_size, static_channels].
        Returns:
            A tuple containing:
            - The output tensor after applying conditional feature mixing.
            - The transformed static features tensor for monitoring or further processing.
        """
        v = self.fr_static(x_static)  # Transform static features to match output channels.
        v = v.unsqueeze(1).repeat(1, x.shape[1], 1)  # 将静态特征扩展并重复以匹配动态特征的时间步长，使其在整个序列长度上保持一致。
        return (self.fm(torch.cat([x, v], dim=-1)),  #  将扩展的静态特征和动态特征沿最后一个维度（特征维度）合并,对合并后的特征应用 FeatureMixing 模块
               v.detach(),  # Return detached static feature for monitoring or further use.
        )

#时间融合
class TimeMixing(nn.Module ):
    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = nn.BatchNorm1d,  # Updated to use a proper 1D norm
        # norm_type: type[nn.Module] = TimeBatchNorm2d,  # Updated to use a proper 1D norm
    ):
        super(TimeMixing, self).__init__()
        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(sequence_length, sequence_length)

        self.attention = LiteMLA(in_channels=input_channels, out_channels=input_channels, heads=2, dim=4)

    def forward(self, x):
        # 将输入转换为时间维度
        x_temp = feature_to_time(x)
        x_temp = self.fc1(x_temp)
        x_temp = self.activation_fn(x_temp)
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)
        # 注意力机制
        x_attention = self.attention(x_res)
        x = x + x_attention
        # 应用归一化并与原始输入相结合
        return self.norm(x)


class MixerLayer(nn.Module):
    """A residual block that combines time and feature mixing for sequence data.

    This module sequentially applies time mixing and feature mixing, which are forms
    of data augmentation and feature transformation that can help in learning temporal
    dependencies and feature interactions respectively.

    Args:
        sequence_length: The length of the input sequences.
        input_channels: The number of input channels to the module.
        output_channels: The number of output channels from the module.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both time and feature mixing.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        """Initializes the MixLayer with time and feature mixing modules."""
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = FeatureMixing(
            sequence_length,
            input_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
            normalize_before=normalize_before,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MixLayer module.

        Args:
            x: A 3D tensor with shape (N, C, L) to be processed by the mixing layers.

        Returns:
            The output tensor after applying time and feature mixing operations.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x = self.feature_mixing(x)  # Then apply feature mixing.

        return x


class ConditionalMixerLayer(nn.Module):
    """Conditional mix layer combining time and feature mixing with static context.

    This module combines time mixing and conditional feature mixing, where the latter
    is influenced by static features. This allows the module to learn representations
    that are influenced by both dynamic and static features.

    Args:
        sequence_length: The length of the input sequences.
        input_channels: The number of input channels of the dynamic features.
        output_channels: The number of output channels after feature mixing.
        static_channels: The number of channels in the static feature input.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both mixing operations.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        static_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = ConditionalFeatureMixing(
            sequence_length,
            input_channels,
            output_channels=output_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """Forward pass for the conditional mix layer.

        Args:
            x: A tensor representing dynamic features, typically with shape
               [batch_size, time_steps, input_channels].
            x_static: A tensor representing static features, typically with shape
               [batch_size, static_channels].

        Returns:
            The output tensor after applying time and conditional feature mixing.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x, _ = self.feature_mixing(x, x_static)  # Then apply conditional feature mixing.

        return x


def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series tensor to a feature tensor."""
    return x.permute(0, 2, 1)

feature_to_time = time_to_feature

#傅里叶变换
class FFTProcessor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        """
        初始化 FFTProcessor 类，设置全连接层、ReLU激活函数和Dropout层。
        
        参数:
        - input_size: 输入数据的最后一维大小 (即 channels)
        - dropout_rate: Dropout 层的丢弃概率
        """
        super(FFTProcessor, self).__init__()
        self.fc_layer = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_hist):
        """
        对输入数据进行傅里叶变换，实部和虚部分别经过全连接层、ReLU激活和Dropout处理，
        然后进行残差连接，最后进行逆傅里叶变换返回时域。

        参数:
        - x_hist: 输入数据 (batch_size, seq_len, channels)

        返回:
        - x_his_final: 经过处理后返回时域的最终结果
        """
        # 对 x_hist 进行傅里叶变换
        x_his_fft = torch.fft.fft(x_hist, dim=-1)
        # 分别处理实部和虚部
        real_part = self.dropout(self.relu(self.fc_layer(x_his_fft.real)))
        imag_part = self.dropout(self.relu(self.fc_layer(x_his_fft.imag)))
        # 合并处理后的实部和虚部
        x_his_fft_final = real_part + imag_part * 1j
        # 残差连接
        x_his_fft_residual = x_his_fft + x_his_fft_final
        # 进行逆傅里叶变换
        x_his_final = torch.fft.ifft(x_his_fft_residual, dim=-1).real
        return x_his_final


#STAR模块
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
