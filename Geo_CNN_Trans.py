import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, auc


# 自定义 Dataset 类（包含中心点坐标）
class VoxelDataset(Dataset):
    def __init__(self, data_dir):
        # 直接读取单个数据文件
        data_file = os.path.join(data_dir, "data.npy")
        label_file = os.path.join(data_dir, "label.npy")
        mask_file = os.path.join(data_dir, "mask.npy")
        center_file = os.path.join(data_dir, "center.npy")

        # 加载数据
        self.data = np.load(data_file)
        self.labels = np.load(label_file)
        self.labels[self.labels == -1] = 0
        self.masks = np.load(mask_file)
        self.centers = np.load(center_file)

        print(f"加载数据：{self.data.shape[0]} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        m = torch.tensor(self.masks[idx], dtype=torch.float32)
        center = self.centers[idx]  # (x, y, z) voxel 坐标
        return x, m, y, center


# 加载数据
def load_data(data_dir, batch_size, test_ratio=0.2):
    train_data = VoxelDataset(data_dir)
    test_size = int(len(train_data) * test_ratio)
    train_size = len(train_data) - test_size
    train_dataset, test_dataset = random_split(train_data, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 空间感知位置编码（X, Y, Z 三方向 Embedding）
class SpatialPositionalEncoding3D(nn.Module):
    def __init__(self, depth, height, width, dim):
        super().__init__()
        self.z_embed = nn.Embedding(depth, dim)
        self.y_embed = nn.Embedding(height, dim)
        self.x_embed = nn.Embedding(width, dim)

    def forward(self, B, D, H, W, device):
        z = torch.arange(D, device=device).view(D, 1, 1).expand(D, H, W)
        y = torch.arange(H, device=device).view(1, H, 1).expand(D, H, W)
        x = torch.arange(W, device=device).view(1, 1, W).expand(D, H, W)
        pos = self.z_embed(z) + self.y_embed(y) + self.x_embed(x)
        pos = pos.view(-1, pos.shape[-1])
        pos = pos.unsqueeze(0).expand(B, -1, -1)
        return pos


# 主模型结构：3D-CNN + Transformer
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.relu

        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask,
                                                   need_weights=True, average_attn_weights=False)
        self.attn_weights = attn_weights  # [B, heads, N, N]
        src = src + attn_output
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm2(src)
        return src


class CNNTransformer3D(nn.Module):
    def __init__(self, in_channels=15, cnn_dim=32, transformer_dim=32,
                 nhead=4, num_layers=2, depth=9, height=9, width=9):
        super().__init__()

        # CNN部分（提取局部空间特征）
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),

        )


        self.pos_enc = SpatialPositionalEncoding3D(
            depth=depth, height=height, width=width, dim=transformer_dim
        )


        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(transformer_dim, nhead) for _ in range(num_layers)
        ])


        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, mask):
        cnn_feat = self.cnn(x)
        B, C, D, H, W = cnn_feat.shape
        device = cnn_feat.device

        tokens = cnn_feat.flatten(2).transpose(1, 2)
        pos = self.pos_enc(B, D, H, W, device=device)
        tokens = tokens + pos

        for layer in self.transformer_layers:
            tokens = layer(tokens)
        encoded = tokens

        mask = (mask > 0.5).float()
        mask = mask.view(B, -1).float()
        mask_sum = mask.sum(dim=1, keepdim=True) + 1e-6
        mask = mask / mask_sum

        pooled = torch.bmm(mask.unsqueeze(1), encoded).squeeze(1)
        logits = self.classifier(pooled).squeeze(-1)



        return logits, cnn_feat


# 元素分带损失函数
class ZoningAwareLoss(nn.Module):
    def __init__(self, base_loss=nn.BCEWithLogitsLoss(), lambda_zone=0.3):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_zone = lambda_zone

    def _spatial_continuity(self, features, masks):

        B, C, D, H, W = features.shape

        # 计算三维梯度
        grad_x = features[:, :, :, 1:, :] - features[:, :, :, :-1, :]
        grad_y = features[:, :, :, :, 1:] - features[:, :, :, :, :-1]
        grad_z = features[:, :, 1:, :, :] - features[:, :, :-1, :, :]

        # 应用掩码
        mask_x = masks[:, :, :, 1:, :] * masks[:, :, :, :-1, :]
        mask_y = masks[:, :, :, :, 1:] * masks[:, :, :, :, :-1]
        mask_z = masks[:, :, 1:, :, :] * masks[:, :, :-1, :, :]

        # 计算W距离
        wd = (torch.mean(torch.abs(grad_x * mask_x)) +
              torch.mean(torch.abs(grad_y * mask_y)) +
              torch.mean(torch.abs(grad_z * mask_z))) / 3.0
        return wd

    def forward(self, logits, targets, features, masks):
        main_loss = self.base_loss(logits, targets)
        zone_loss = self._spatial_continuity(features, masks)
        return main_loss + self.lambda_zone * zone_loss



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for x, mask, y, _ in dataloader:  # 加载 mask
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        logits, cnn_feat = model(x, mask)  # 将 mask 传入模型
        loss = criterion(logits, y, cnn_feat, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    return avg_loss, accuracy



def test_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    y_true, y_prob, y_pred = [], [], []

    with torch.no_grad():
        for x, mask, y, _ in dataloader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits, cnn_feat = model(x, mask)
            loss = criterion(logits, y, cnn_feat, mask)

            total_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()

            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    return avg_loss, accuracy, y_true, y_prob, y_pred


# 主训练函数
def train_model(model, train_loader, test_loader, epochs, criterion, optimizer, tol, PATH, modelname, device):
    best_acc = 0
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, y_true, y_prob, y_pred = test_one_epoch(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # 计算roc和指标
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"\tTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"\tTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"\tPrecision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}, AUC: {roc_auc:.4f}")


        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(PATH, modelname + ".pt"))
            print("  Best model updated and saved.")

    return train_losses, test_losses, fpr, tpr


def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Testing Loss")
    plt.grid(True)
    plt.show()


def plot_roc_curve(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


trainloader, testloader = load_data(r"F:\data_0524\train_data", batch_size=64)


model = CNNTransformer3D(in_channels=11).to(device)


criterion = ZoningAwareLoss(base_loss=nn.BCEWithLogitsLoss(), lambda_zone=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# 训练模型
save_path = r"F:\data_0524\train_data"  # 请确保该路径存在
model_name = "cnn_transformer3d"

trainloss, testloss, fpr, tpr = train_model(
    model, trainloader, testloader,
    epochs=30,
    criterion=criterion,
    optimizer=optimizer,
    tol=1e-4,
    PATH=save_path,
    modelname=model_name,
    device=device
)

# 可视化
plot_loss(trainloss, testloss)
plot_roc_curve(fpr, tpr)

# 循环完毕后，清理GPU内存
print("训练完成，清理缓存...")
del model
torch.cuda.empty_cache()
gc.collect()
