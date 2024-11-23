import pandas as pd
import numpy as np
import sys
import time
import datetime
from sklearn.preprocessing import StandardScaler
from torchtsmixer.tsmixer_ext import TSMixerExt
from torchtsmixer.tsmixer_ext2 import TSMixerExt2
import torch
import torch.nn as nn
import torch.optim as optim
import os

#print_my函数用于打印日志信息，同时将日志信息写入文件。首次调用时会创建一个日志文件，并以后续调用时继续向该文件追加内容。
def print_my(message):
    if not hasattr(print_my, 'initialized'):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print_my.current_filename = f"./log/output_{current_time}.log"
        print_my.initialized = True
        print(f"Logging to file: {print_my.current_filename}")
    print(message)
    with open(print_my.current_filename, 'a') as f:
        print(message, file=f)


def read_and_prepare_data(seq_len, pred_len, input_slice):
    folder_path = 'data/dataset'
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    #读取静态特征
    file_path = 'data/static.xlsx'
    df_static_features = pd.read_excel(file_path, header=0)
    # 去除前三列
    df_static_features = df_static_features.drop(df_static_features.columns[:3], axis=1)

    scaler = StandardScaler()
    scaler.fit(df_static_features.values)
    def scale_df(df, scaler):
        data = scaler.transform(df.values)
        return pd.DataFrame(data, index=df.index, columns=df.columns)
    df_static_features = scale_df(df_static_features, scaler)

    data_train_x_hist = []
    data_train_x_static = []
    data_train_y = []
    data_test_x_hist = []
    data_test_x_static = []
    data_test_y = []
    for index, file in enumerate(files):
        df = pd.read_excel(file, header=0)
        # print(df.columns)
        #修改数据的列名，替换原名
        df.rename(columns={'测量时间': 'data'}, inplace=True)
        df = df.set_index('data') #设置索引
        # print(df.columns)
        n = len(df)
        train_end = int(n * 0.8)
        test_end = n
        df_train = df[:train_end]
        df_test = df[train_end: test_end]
        scaler = StandardScaler()
        scaler.fit(df_train.values)
        # def scale_df(df, scaler):
        #     data = scaler.transform(df.values)
        #     return pd.DataFrame(data, index=df.index, columns=df.columns)
        df_train = scale_df(df_train, scaler)
        df_test = scale_df(df_test, scaler)
        #创建时间序列数据的滑动窗口
        tmp_train = []
        features_train = df_train.values
        for i in range(len(df_train) - seq_len - pred_len + 1):
            tmp_train.append(features_train[i:i + seq_len + pred_len])
        data_train = np.array(tmp_train)
        # print(df_train)
        # print(f"data_train shape: {data_train.shape}")
        # print(df_train.columns)  # 打印所有列
        # print(df_train.index)  
        # 对应的张量划分
        single_train_x_hist = torch.tensor(data_train[:, :seq_len, input_slice])
       
        # 获取对应的静态特征
        static_feat1 = df_static_features.iloc[index].values
        static_feat_tensor1 = torch.tensor(static_feat1, dtype=torch.float32).unsqueeze(0) 
        single_train_x_static = static_feat_tensor1.repeat(data_train.shape[0],1)
        single_train_y = torch.tensor(data_train[:, seq_len:, output_slice])

        data_train_x_hist.append(single_train_x_hist)
        data_train_x_static.append(single_train_x_static)
        data_train_y.append(single_train_y)
        
        # 构建测试序列
        tmp_test = []
        features_test = df_test.values
        if len(features_test) >= seq_len + pred_len:
            for i in range(len(features_test) - seq_len - pred_len + 1):
                tmp_test.append(features_test[i:i + seq_len + pred_len])
            data_test = np.array(tmp_test)

            single_test_x_hist = torch.tensor(data_test[:, :seq_len, input_slice])
            
            static_feat2 = df_static_features.iloc[index].values
            static_feat_tensor2 = torch.tensor(static_feat2, dtype=torch.float32).unsqueeze(0) 
            single_test_x_static = static_feat_tensor2.repeat(data_test.shape[0], 1)
            single_test_y = torch.tensor(data_test[:, seq_len:, output_slice])

            data_test_x_hist.append(single_test_x_hist)
            data_test_x_static.append(single_test_x_static)
            data_test_y.append(single_test_y)
        else:
            print("Not enough data to create sequences")

    data_train_x_hist = torch.cat(data_train_x_hist, dim=0)
    data_train_x_static = torch.cat(data_train_x_static, dim=0)
    data_train_y = torch.cat(data_train_y, dim=0)

    data_test_x_hist = torch.cat(data_test_x_hist, dim=0)
    data_test_x_static = torch.cat(data_test_x_static, dim=0)
    data_test_y = torch.cat(data_test_y, dim=0)
    
    return data_train_x_hist,data_train_x_static,data_train_y,data_test_x_hist,data_test_x_static,data_test_y


data_type = 'diabetes'  # solar wind
if  data_type == 'diabetes':
    print_my('data type: diabetes')

    seq_len = 9 #时间序列长度，前两个小时的数据
    pred_len = 8 #预测长度。预测后两个小时的数据
    input_channels = 2 #添加傅里叶变换之后，通道数变为2
    hidden_channels = 8 #任意
    static_channels = 50
    output_channels = 1
    input_slice = [1] #输入数据中使用的特征列
    output_slice = [1] #输出数据中使用的特征列
    # 训练超参数
    num_epochs = 300
    batch_size = 512
    learning_rate = 0.001  # 0.0001
    
    data_train_x_hist,data_train_x_static,data_train_y,data_test_x_hist,data_test_x_static,data_test_y = read_and_prepare_data(seq_len, pred_len, input_slice)
    # print(data_train_x_hist.shape)
    # print(data_train_x_static.shape)
    # print(data_train_y.shape)
else:
    print_my('data_type is not support')
    sys.exit(1)#检测到错误或异常条件时，立即停止程序执行，并且向外界报告这种错误状态。

print_my('param is:')
print_my('seq_len = ' + str(seq_len))
print_my('pred_len = ' + str(pred_len))
print_my('input_channels = ' + str(input_channels))
print_my('hidden_channels = ' + str(hidden_channels))
print_my('static_channels = ' + str(static_channels))
print_my('output_channels = ' + str(output_channels))
print_my('input_slice = ' + str(input_slice))
print_my('output_slice = ' + str(output_slice))
print_my('num_epochs = ' + str(num_epochs))
print_my('batch_size = ' + str(batch_size))
print_my('learning_rate = ' + str(learning_rate))

# 构建模型
model = TSMixerExt2(
    sequence_length=seq_len,
    prediction_length=pred_len,
    input_channels=input_channels,
    hidden_channels=hidden_channels,
    static_channels=static_channels,
    output_channels=output_channels
)

model.cuda()
# 设置损失函数和优化器
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 初始化早停相关变量
best_mse = float('inf')
best_mae = float('inf')
patience = 10
trigger_times = 0

# 模型训练&预测
for epoch in range(num_epochs):
    # 训练
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for batch in range(0, data_train_x_hist.shape[0], batch_size):
        x1 = data_train_x_hist[batch: batch + batch_size]
        x4 = data_train_x_static[batch: batch + batch_size]
        y = data_train_y[batch: batch + batch_size]
        # 模型前向传播和损失计算
        optimizer.zero_grad()
        outputs = model.forward(
            x_hist=x1.float().cuda(),
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
        x_static=data_test_x_static.float().cuda()
    )
    pred_mse = criterion_mse(y_pred, data_test_y.float().cuda())
    pred_mae = criterion_mae(y_pred, data_test_y.float().cuda())
    print_my(f'Epoch [{epoch + 1}/{num_epochs}], MSE: {pred_mse.item():.4f}, MAE: {pred_mae.item():.4f}\n')

    # 早停策略
    if pred_mse < best_mse:
        best_mse = pred_mse
        best_mae = pred_mae
        trigger_times = 0  # 重置计数器
        # 保存模型权重
        torch.save(model.state_dict(), 'model_file/CC_9_8.pth')
    else:
        trigger_times += 1
        print_my(f'EarlyStopping counter: {trigger_times} out of {patience}')
        print_my(f'*************')
        if trigger_times >= patience:
            print_my(f'Early stopping triggered. Stopping training.')
            break

# 如果训练结束时有最佳模型，加载它并打印最终的MSE和MAE
model.load_state_dict(torch.load('model_file/CC_9_8.pth'))
model.eval()
y_pred = model.forward(
    x_hist=data_test_x_hist.float().cuda(),
    x_static=data_test_x_static.float().cuda()
)
final_mse = criterion_mse(y_pred, data_test_y.float().cuda())
final_mae = criterion_mae(y_pred, data_test_y.float().cuda())
print_my(f'Final Best Model Evaluation -> MSE: {final_mse.item():.4f}, MAE: {final_mae.item():.4f}')

