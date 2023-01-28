import configparser
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Linear

from dataset import MyDataset
from sampler import RatioDatasetSampler
from util import float_to_percent


class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mlp = nn.Sequential(Linear(128, 64, weight_initializer='kaiming_uniform'),
                                 nn.LeakyReLU(),
                                 Linear(64, 2, weight_initializer='kaiming_uniform'))
        self.sm = nn.Softmax()

    def forward(self, data):
        x = data.x
        h = self.mlp(x)
        out = self.sm(h)
        return out


def idx2index(idx: torch.Tensor) -> torch.Tensor:
    """
    根据稀疏矩阵求index
    """
    index = []
    size = idx.shape[0]
    for i in range(size):
        if idx[i].item() == 1:
            index.append(i)
    return torch.tensor(index).long()


def visual(x, y, epoch):
    # t-SNE的最终结果的降维与可视化
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, )
    # x = tsne.fit_transform(x)
    # x_min, x_max = np.min(x, 0), np.max(x, 0)
    # x = (x - x_min) / (x_max - x_min)

    color = ['lightblue', 'red']
    for i in range(x.shape[0]):
        plt.text(x[i, 0],
                 x[i, 1],
                 str(y[i]),
                 color=color[y[i]])
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    f = plt.gcf()  # 获取当前图像
    f.savefig(f'./result/{epoch}.png')
    f.clear()  # 释放内存
    plt.show()


if __name__ == '__main__':
    """
    这个文件实现了最简单的模型
    我们对每个AST求和平均并直接使用MLP进行二分类预测
    目标项目为zookeeper
    """

    # 第一步 读取训练集
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    project = cf.get('data', 'projectName')

    root_dir = cf.get('data', 'dataDir')
    root_dir = os.path.join(root_dir, project)

    project = 'zookeeperdemo'
    BS = 32
    LR = 1e-4
    EPOCHS = 15

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = MyDataset(root=root_dir, project=project, dataset_type="train")
    val_dataset = MyDataset(root=root_dir, project=project, dataset_type="val")
    methods_info = pd.read_pickle(os.path.join(root_dir, 'processed', 'method_info.pkl'))


    def my_collate_fn(batch):
        new_datalist = []
        for data in batch:
            method = data.id.split('@')[0]
            info = methods_info.loc[methods_info['id'] == method]

            x = info['ASTs'].tolist()[0]
            x = x[idx2index(data.idx).item()].x
            x = x.mean(axis=0)
            x = x.reshape(1, 128)
            y = torch.tensor([[0, 1]]) if data.y.item() == 0 else torch.tensor([[1, 0]])

            new_data = Data(
                id=data.id,
                x=x,
                y=y,
            )

            new_datalist.append(new_data)

        return Batch.from_data_list(new_datalist)


    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=my_collate_fn,
                              sampler=RatioDatasetSampler(train_dataset, 3),
                              batch_size=BS,
                              shuffle=False)

    val_loader = DataLoader(dataset=val_dataset,
                            collate_fn=my_collate_fn,
                            batch_size=BS,
                            shuffle=False)

    model = MyMLP().to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_function = torch.nn.BCELoss().to(device)

    # 定义用于评估预测结果的东西
    best_acc = 0.0
    best_model = model

    # 定义控制日志打印的东西
    total_train_step = 0
    for epoch in range(EPOCHS):
        print(f'------------第 {epoch + 1} 轮训练开始------------')
        model.train()
        for i, data in enumerate(train_loader):
            y_hat = model(data.to(device))
            y = data.y.float()
            loss = loss_function(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print(f"训练次数: {total_train_step}, Loss: {loss.item()}")

        # 开始验证！
        total_val_loss = 0.0
        y_hat_total = torch.randn(0).to(device)
        y_total = torch.randn(0).to(device)

        ys_neg = []
        xs_neg = torch.randn(0, 2)

        ys_pos = []
        xs_pos = torch.randn(0, 2)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                y_hat = model(data.to(device))
                y = data.y.float()
                loss = loss_function(y_hat, y)

                # 用来计算整体指标
                total_val_loss += loss.item()
                y_hat_trans = y_hat.argmax(1)
                y_trans = y.argmax(1)
                y_hat_total = torch.cat([y_hat_total, y_hat_trans])
                y_total = torch.cat([y_total, y_trans])

                for i in range(len(data)):
                    item = data[i]
                    if item.y.equal(torch.tensor([[0, 1]])):
                        xs_neg = torch.cat([xs_neg, torch.index_select(y_hat, dim=0, index=torch.tensor([i]))], dim=0)
                        ys_neg.append(0)
                    else:
                        xs_pos = torch.cat([xs_neg, torch.index_select(y_hat, dim=0, index=torch.tensor([i]))], dim=0)
                        ys_pos.append(1)

        print(f"验证集整体Loss: {total_val_loss}")
        acc = accuracy_score(y_total.cpu(), y_hat_total.cpu())
        balanced_acc = balanced_accuracy_score(y_total.cpu(), y_hat_total.cpu())
        ps = precision_score(y_total.cpu(), y_hat_total.cpu())
        rc = recall_score(y_total.cpu(), y_hat_total.cpu())
        f1 = f1_score(y_total.cpu(), y_hat_total.cpu())
        c = confusion_matrix(y_total.cpu(), y_hat_total.cpu(), labels=[0, 1])

        print(f"验证集 accuracy_score: {float_to_percent(acc)}")
        print(f"验证集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
        print(f"验证集 precision_score: {float_to_percent(ps)}")
        print(f"验证集 recall_score: {float_to_percent(rc)}")
        print(f"验证集 f1_score: {float_to_percent(f1)}")
        print(f"验证集 混淆矩阵:\n {c}")

        if balanced_acc > best_acc:
            print(f"***当前模型的平衡准确率表现最好，被记为表现最好的模型***\n")
            best_model = model
            best_acc = balanced_acc

        print(f"***保存tsne中***\n")
        ys = []
        ys += ys_neg
        ys += ys_pos
        ys = np.array(ys)

        xs = torch.cat([xs_neg, xs_pos], dim=0)
        xs = xs.numpy()

        visual(xs, ys, epoch + 1)
