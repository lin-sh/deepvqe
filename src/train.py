from deepvqe_v1 import DeepVQE
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


class CSVAudioDataset(Dataset):
    def __init__(self, csv_file):
        # 加载CSV文件
        self.dataframe = pd.read_csv(csv_file, header=None)
        self.far_end_paths = sorted(self.dataframe[0].values)

    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        # 加载lpb和mic音频文件
        farend_signal, sample_rate = sf.read(self.far_end_paths[idx])
        
        return torch.tensor(farend_signal).float()

def validate(model, val_loader, criterion, device):
    model.eval()  # set the model to evaluation mode
    total_val_loss = 0.0
    num = 2160

    with torch.no_grad():  # no need to compute gradients during validation
        for batch_idx, (lpb, mic) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", leave=False):
            lpb, mic = lpb.to(device), mic.to(device)
            # 使用模型进行预测。注意，这里的forward方法应该处理lpb和mic的组合来生成预测。
            lpb, mic = pad_to_match(lpb, mic)
            prediction = model(lpb, mic)
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(0)
            prediction, mic = pad_to_match(prediction, mic)
            if torch.isnan(prediction).any():
                num-=1
            # print(num)
            continue
            # 计算损失
            loss = criterion(prediction, mic)  # 假设mic是你想预测的目标
            total_val_loss += loss.item()

    return total_val_loss / len(val_loader)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num = 10000
    total_batches = len(train_loader)
    # 使用tqdm来显示进度条
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    
    for batch_idx, (far_end, near_end, echo) in pbar:
        # 将数据移到指定的设备
        far_end, near_end, near_echo = far_end.to(device), near_end.to(device), echo.to(device)
        optimizer.zero_grad()  # 清除梯度
        echo_hat = model(far_end, near_echo)  # 获取模型预测
        
        if echo_hat.dim() == 1:
            echo_hat = echo_hat.unsqueeze(0)
        if torch.isnan(echo_hat).any():
            print('echo_hat has nan')
            nan_count = torch.sum(torch.isnan(echo_hat)).item()
            print(f"Number of NaN values: {nan_count}")

        # 计算损失
        loss = criterion(echo_hat, far_end)
        loss.backward()
        optimizer.step()  # 更新参数
        
        # 将损失加到总损失中
        total_loss += loss.item()
        
        # 在tqdm进度条中显示当前的损失
        pbar.set_description(f"Training Loss: {loss.item():.4f}")

    return total_loss / total_batches, num

def setup_logging():
    logging.basicConfig(filename="log/training.log", level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    # 如果你还想在控制台中看到这些日志，添加以下代码：
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def main(args):
    # 初始化数据集和数据加载器
    dataset = CSVAudioDataset(args.speech_corpus_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 验证集
    # val_dataset = CSVAudioDataset(args.val_corpus_path)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    best_loss = float('inf')

    # 设定硬件设备，定义模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NKF(L=4, train=True).to(device)
    # criterion = CombinedLoss(alpha=10000, device = device)
    # criterion = ComplexMAELoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_history = []  # 保存每个epoch的损失

    setup_logging()

    # 主训练循环
    for epoch in range(args.epochs):
        loss, num = train(model, train_loader, criterion, optimizer, device)
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
        print(num)
        loss_history.append(loss)

        # # Validate every epoch (you can change this to validate every N epochs)
        # val_loss = validate(model, val_loader, criterion, device)
        # logging.info(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.4f}")


        # Check if this is the best model so far
        if loss < best_val_loss:
            logging.info(f"New best model found! Saving to 'best_model.pth'")
            best_val_loss = loss
            torch.save(model.state_dict(), "/root/NKF-AEC/src/model/best_model_"+epoch+".pt")

    # 在训练结束后，绘制损失的变化图
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("/root/NKF-AEC/src/model/loss.png")

if __name__ == "__main__":
    # 使用argparse处理命令行参数
    parser = argparse.ArgumentParser(description="NKF model training")
    parser.add_argument('--speech-corpus-path', type=str, default = r"/root/aec/NKF-AEC/src/AEC-Challenge/datasets/synthetic/python/far-end/far-end-train.csv", help='Path to the synthetic speech corpus')
    parser.add_argument('--val_corpus_path', type=str, default = r"/root/aec/NKF-AEC/src/AEC-Challenge/datasets/synthetic/python/far-end/far-end-val.csv", help='Path to the synthetic speech corpus')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    if len(sys.argv[1:])==0:
        args = parser.parse_args(args=[])  # 获取参数
    else:
        args = parser.parse_args() 

    main(args)
