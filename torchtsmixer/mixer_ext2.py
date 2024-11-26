from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers2 import (
    ConditionalFeatureMixing,
    ConditionalMixerLayer,
    TimeBatchNorm2d,
    FFTProcessor,
    STAR,
    feature_to_time,#在特征维度和时间维度之间进行转换
    time_to_feature,
)

class TSMixerExt2(nn.Module):
    """TSMixer model for time series forecasting.
        This model forecasts time series data by integrating historical time series data,
    future known inputs, and static contextual information. It uses a combination of
    conditional feature mixing and mixer layers to process and combine these different
    types of data for effective forecasting.
    Args:
        sequence_length: The length of the input time series sequences.
        prediction_length: The length of the output prediction sequences.
        activation_fn: The name of the activation function to be used.
        num_blocks: The number of mixer blocks in the model.
        dropout_rate: The dropout rate used in the mixer layers.
        input_channels: The number of channels in the historical time series data.
        extra_channels: The number of channels in the extra (future known) inputs.
        hidden_channels: The number of hidden channels used in the mixer layers.
        static_channels: The number of channels in the static feature inputs.
        ff_dim: The inner dimension of the feedforward network in the mixer layers.
        output_channels: The number of output channels for the final output. If None,
                         defaults to the number of input_channels.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: The type of normalization to use. "batch" or "layer".
    """

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        activation_fn: str = "relu",
        num_blocks: int = 4,
        dropout_rate: float = 0.5,
        input_channels: int = 1,
        hidden_channels: int = 8,
        static_channels: int = 50,
        ff_dim: int = 64,
        output_channels: int = None,
        normalize_before: bool = False,
        norm_type: str = "layer",
        input_size: int = 1,
        d_series: int = 1,
        d_core: int = 16
    ):
        #检查static_channels必须大于0，保证有静态特征输入。
        assert static_channels > 0, "static_channels must be greater than 0"
        super().__init__()

        # 检查激活函数字符串是否对应于 torch.nn.functional (即 F) 中的一个有效函数，如果是，将字符串转换为该函数的实际可调用形式。如果不是，抛出一个错误。
        if hasattr(F, activation_fn):
            activation_fn = getattr(F, activation_fn)
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

        # 检查norm_type是否有效，确保它是"batch"或"layer"中的一个。然后根据这个字符串选择相应的归一化类
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm
        
        #创建一个线性层，用于将历史序列数据从 sequence_length 转换到 prediction_length。
        self.fc_hist = nn.Linear(sequence_length, prediction_length)
        #创建一个输出线性层，用于将最终的隐藏层表示转换为输出通道。如果没有明确指定output_channels，则默认使用input_channels
        self.fc_out = nn.Linear(hidden_channels, output_channels or input_channels)
        #傅里叶变换
        self.fftprocessor = FFTProcessor(input_size=input_size, dropout_rate=0.5)
        #STAR模块
        self.star = STAR(d_series=d_series, d_core=d_core)

        self.feature_mixing_hist = ConditionalFeatureMixing(
            sequence_length=prediction_length,
            # input_channels=input_channels + extra_channels,
            input_channels=input_channels,
            output_channels=hidden_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        
        self.conditional_mixer = self._build_mixer(
            num_blocks,
            hidden_channels,
            prediction_length,
            ff_dim=ff_dim,
            static_channels=static_channels,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    @staticmethod
    def _build_mixer(
        num_blocks: int, hidden_channels: int, prediction_length: int, **kwargs
    ):
        """Build the mixer blocks for the model."""
        channels = [hidden_channels] + [hidden_channels] * (num_blocks - 1)
        return nn.ModuleList(
            [
                ConditionalMixerLayer(
                    input_channels=in_ch,
                    output_channels=out_ch,
                    sequence_length=prediction_length,
                    **kwargs,
                )
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )
    
    def forward(
        self,
        x_hist: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the TSMixer model.
        Processes historical and future data, along with static features, to produce a forecast.
        Args:
            x_hist: Historical time series data (batch_size, sequence_length, input_channels).
            x_extra_hist: Additional historical data (batch_size, sequence_length, extra_channels).
            x_extra_future: Future known data (batch_size, prediction_length, extra_channels).
            x_static: Static contextual data (batch_size, static_channels).
        Returns:
            The output tensor representing the forecast (batch_size, prediction_length, output_channels).
        """
        # Concatenate historical time series data with additional historical data
        # print('*' * 88)
        # print(x_hist.shape)  # 480——11
        x_hist1 = x_hist.clone()
        x_hist1 = self.fftprocessor(x_hist1)#傅里叶变换
        # print(x_hist1.shape)
        x_hist = self.star(x_hist, x_hist1)

        # print(x_hist.shape)
        # #Transform feature space to time space, apply linear trafo, and convert back
        x_hist_temp = feature_to_time(x_hist)
        x_hist_temp = self.fc_hist(x_hist_temp)
        x_hist = time_to_feature(x_hist_temp)
        # #print(x_hist.shape)  # 96——11
        # #print(x_static.shape)
        # #Apply conditional feature mixing to the historical data
        x_hist, _ = self.feature_mixing_hist(x_hist, x_static=x_static)
        # #print(x_hist.shape)  # 96——8

        # #Apply conditional feature mixing to the future data
        # #x_future, _ = self.feature_mixing_future(x_extra_future, x_static=x_static)

        # Concatenate processed historical and future data
        # x = torch.cat([x_hist, x_future], dim=-1)
        # x1 = torch.cat([x_hist, x_hist], dim=-1)
        # print(x_hist.shape)  # 96——8
        # print(x_future.shape)  # 96——8
        # print(x.shape)  # 96——16
        
        # Process the concatenated data through the mixer layers
        for mixing_layer in self.conditional_mixer:
            x = mixing_layer(x_hist, x_static=x_static)

        # Final linear transformation to produce the forecast
        x = self.fc_out(x)

        return x


