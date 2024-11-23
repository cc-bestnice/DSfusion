import pandas as pd
import numpy as np
import sys
import time
import datetime
from sklearn.preprocessing import StandardScaler
from torchtsmixer import TSMixerExt
import torch
import torch.nn as nn
import torch.optim as optim

#print_my函数用于打印日志信息，同时将日志信息写入文件。首次调用时会创建一个日志文件，并以后续调用时继续向该文件追加内容。
def print_my(message):
    # print_my函数首次执行时，它不会有一个名为 'initialized' 的属性，为真时，
    # 代码设置了当前时间，创建了一个日志文件名，并将这个文件名存储为函数的一个属性
    if not hasattr(print_my, 'initialized'):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print_my.current_filename = f"./log/output_{current_time}.log"
        print_my.initialized = True
        print(f"Logging to file: {print_my.current_filename}")
    print(message)
    with open(print_my.current_filename, 'a') as f:
        print(message, file=f)

# data_type = 'solar'  # solar wind
data_type = 'wind'  # solar wind

if data_type == 'solar':
    df = pd.read_excel('data/Solar station site 2 (Nominal capacity-130MW).xlsx', header=0)
    df.rename(columns={'Time(year-month-day h:m:s)': 'data'}, inplace=True)
    print_my('data type: solar')
    # 定义时间序列长度
    seq_len = 480
    pred_len = 96
    input_channels = 1
    extra_channels = 5
    hidden_channels = 8
    static_channels = 1
    output_channels = 1
    input_slice = [5]
    extra_slice = [0, 1, 2, 3, 4]
    output_slice = [5]
    # 训练超参数
    num_epochs = 2000
    batch_size = 128
    learning_rate = 0.0001  # 0.0001

elif data_type == 'wind':
    df = pd.read_excel('data/Wind farm site 2 (Nominal capacity-200MW).xlsx', header=0)
    #修改数据的列名，替换原名
    df.rename(columns={'Time(year-month-day h:m:s)': 'data'}, inplace=True)
    print_my('data type: wind')

    seq_len = 480 #时间序列长度
    pred_len = 96 #预测长度
    input_channels = 1 
    extra_channels = 10 
    hidden_channels = 8 #任意？
    static_channels = 1 
    output_channels = 1 
    input_slice = [10] #输入数据中使用的特征列
    extra_slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #额外输入数据中使用的特征列
    output_slice = [10] #输出数据中使用的特征列
    # 训练超参数
    num_epochs = 200
    batch_size = 512
    learning_rate = 0.0001  # 0.0001

elif data_type == 'diabetes':
    df = pd.read_excel('data/766636.xlsx', header=0)
    #修改数据的列名，替换原名
    df.rename(columns={'测量时间': 'data'}, inplace=True)
    print_my('data type: diabetes')

    seq_len = 5 #时间序列长度，前两个小时的数据
    pred_len = 8 #预测长度。预测后两个小时的数据
    input_channels = 1 
    # extra_channels = 0 
    hidden_channels = 8 #任意？
    static_channels = 1 
    output_channels = 1 
    input_slice = [2] #输入数据中使用的特征列
    # extra_slice = [] #额外输入数据中使用的特征列
    output_slice = [2] #输出数据中使用的特征列
    # 训练超参数
    num_epochs = 200
    batch_size = 64
    learning_rate = 0.001  # 0.0001

else:
    print_my('data_type is not support')
    sys.exit(1)#检测到错误或异常条件时，立即停止程序执行，并且向外界报告这种错误状态。

print_my('param is:')
print_my('seq_len = ' + str(seq_len))
print_my('pred_len = ' + str(pred_len))
print_my('input_channels = ' + str(input_channels))
# print_my('extra_channels = ' + str(extra_channels))
print_my('hidden_channels = ' + str(hidden_channels))
print_my('static_channels = ' + str(static_channels))
print_my('output_channels = ' + str(output_channels))
print_my('input_slice = ' + str(input_slice))
# print_my('extra_slice = ' + str(extra_slice))
print_my('output_slice = ' + str(output_slice))
print_my('num_epochs = ' + str(num_epochs))
print_my('batch_size = ' + str(batch_size))
print_my('learning_rate = ' + str(learning_rate))

df = df.set_index('data') #将时间设置为索引
# 划分训练 & 测试集
n = len(df)
train_end = int(n * 0.8)
test_end = n
df_train = df[:train_end]
df_test = df[train_end: test_end]
# 特征归一化
scaler = StandardScaler()
scaler.fit(df_train.values)

def scale_df(df, scaler):
    data = scaler.transform(df.values)
    return pd.DataFrame(data, index=df.index, columns=df.columns)

df_train = scale_df(df_train, scaler)
df_test = scale_df(df_test, scaler)
n_feature = df_train.shape[-1]

"""
    训练
"""
#创建时间序列数据的滑动窗口
tmp = []
features = df_train.values
for i in range(len(df_train) - seq_len - pred_len + 1):
    tmp.append(features[i:i + seq_len + pred_len])
data_train = np.array(tmp)
# print(f"data_train shape: {data_train.shape}")
# 对应的张量划分
data_train_x_hist = torch.tensor(data_train[:, :seq_len, input_slice])
# data_train_x_extra_hist = torch.tensor(data_train[:, :seq_len, extra_slice])  # 历史天气？
# data_train_x_extra_feature = torch.tensor(data_train[:, seq_len:, extra_slice])  # 预测天气？
data_train_x_static = torch.zeros(data_train.shape[0], static_channels, requires_grad=False)
data_train_y = torch.tensor(data_train[:, seq_len:, output_slice])

# 构建测试序列
tmp = []
features = df_test.values
for i in range(len(df_test) - seq_len - pred_len + 1):
    tmp.append(features[i:i + seq_len + pred_len])
data_test = np.array(tmp)

data_test_x_hist = torch.tensor(data_test[:, :seq_len, input_slice])
# data_test_x_extra_hist = torch.tensor(data_test[:, :seq_len, extra_slice])  # 历史天气
# data_test_x_extra_feature = torch.tensor(data_test[:, seq_len:, extra_slice])  # 预测天气
data_test_x_static = torch.zeros(data_test.shape[0], static_channels, requires_grad=False)
data_test_y = torch.tensor(data_test[:, seq_len:, output_slice])

# 构建模型
model = TSMixerExt(
    sequence_length=seq_len,
    prediction_length=pred_len,
    input_channels=input_channels,
    # extra_channels=0,
    hidden_channels=hidden_channels,
    static_channels=static_channels,
    output_channels=output_channels
)

model.cuda()
# 设置损失函数和优化器
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 模型训练&预测
for epoch in range(num_epochs):
    # 训练
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for batch in range(0, data_train_x_hist.shape[0], batch_size):
        x1 = data_train_x_hist[batch: batch + batch_size]
        # x2 = data_train_x_extra_hist[batch: batch + batch_size]
        # x3 = data_train_x_extra_feature[batch: batch + batch_size]
        x4 = data_train_x_static[batch: batch + batch_size]
        y = data_train_y[batch: batch + batch_size]
        # 模型前向传播和损失计算
        optimizer.zero_grad()
        outputs = model.forward(
            x_hist=x1.float().cuda(),
            # x_extra_hist=x2.float().cuda(),
            # x_extra_future=x3.float().cuda(),
            # x_extra_hist=None,
            # x_extra_future=None,
            x_static=x4.float().cuda()
        )
        loss = criterion_mse(outputs, y.float().cuda())
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    # 记录训练损失和时间
    print_my(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, time cost: {time.time() - start_time}')

    # 预测 # 模型评估
    model.eval()
    y_pred = model.forward(
        x_hist=data_test_x_hist.float().cuda(),
        # x_extra_hist=data_test_x_extra_hist.float().cuda(),
        # x_extra_future=data_test_x_extra_feature.float().cuda(),
        # x_extra_hist=None,
        # x_extra_future=None,
        x_static=data_test_x_static.float().cuda()
    )
    pred_mse = criterion_mse(y_pred, data_test_y.float().cuda())
    pred_mae = criterion_mae(y_pred, data_test_y.float().cuda())
    print_my(f'Epoch [{epoch + 1}/{num_epochs}], MSE: {pred_mse.item():.4f}, MAE: {pred_mae.item():.4f}\n')

