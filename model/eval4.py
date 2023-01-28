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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset import MyDataset
from sampler import RatioDatasetSampler
from util import float_to_percent


class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_layers = 1
        self.num_directions = 1  # 单向LSTM
        self.hidden_size = 64

        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True,
                            dropout=0.2)
        self.linear = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)

        output, _ = self.lstm(x, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        pred = self.sig(pred)
        return output[:, -1, :], pred


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
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, )
    x = tsne.fit_transform(x)
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    color = ['lightblue', 'red']
    for i in range(x.shape[0]):
        plt.text(x[i, 0],
                 x[i, 1],
                 str(y[i]),
                 color=color[y[i]])
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    f = plt.gcf()  # 获取当前图像
    if epoch == -1:
        f.savefig(f'./result/test.png')
    else:
        f.savefig(f'./result/{epoch}.png')

    f.clear()  # 释放内存
    plt.show()


if __name__ == '__main__':
    """
    这个文件对eval4中的模型进行了升级
    我们不再是只对AST节点求平均并二分类，而是使用RNN进行对特征进行更新
    因此每一个data包括了一个序列的AST平均值
    目标项目为zookeeper
    """

    # 第一步 读取训练集
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    project = cf.get('data', 'projectName')

    root_dir = cf.get('data', 'dataDir')
    root_dir = os.path.join(root_dir, project)

    project = 'zookeeperdemo'
    BS = 15
    LR = 1e-4
    EPOCHS = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = MyDataset(root=root_dir, project=project, dataset_type="train")
    val_dataset = MyDataset(root=root_dir, project=project, dataset_type="val")
    test_dataset = MyDataset(root=root_dir, project=project, dataset_type="test")
    methods_info = pd.read_pickle(os.path.join(root_dir, 'processed', 'method_info.pkl'))


    def my_collate_fn(batch):
        xs = []
        ys = []
        ids = []

        for data in batch:
            method = data.id.split('@')[0]
            info = methods_info.loc[methods_info['id'] == method]

            x = info['ASTs'].tolist()[0]

            seq = torch.randn(0, 128)
            index = idx2index(data.idx).item()
            for i in range(index + 1):
                ast = x[index].x
                ast = ast.mean(axis=0)
                ast = ast.reshape(1, 128)
                seq = torch.cat([seq, ast], dim=0)

            xs.append(seq)
            ys.append(data.y.item())
            ids.append(data.id)
        xs = pad_sequence(xs, batch_first=True)
        ys = torch.tensor(ys).float()

        return xs, ys, ids


    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=my_collate_fn,
                              sampler=RatioDatasetSampler(train_dataset, 3),
                              batch_size=BS,
                              shuffle=False)

    val_loader = DataLoader(dataset=val_dataset,
                            collate_fn=my_collate_fn,
                            batch_size=BS,
                            shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             collate_fn=my_collate_fn,
                             batch_size=BS,
                             shuffle=False)

    model = MyLSTM().to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_function = torch.nn.MSELoss().to(device)

    # 定义用于评估预测结果的东西
    best_acc = 0.0
    best_model = model

    # 定义控制日志打印的东西
    total_train_step = 0
    for epoch in range(EPOCHS):
        print(f'------------第 {epoch + 1} 轮训练开始------------')
        model.train()
        for i, (x, y, ids) in enumerate(train_loader):
            model.zero_grad()
            _, y_hat = model(x.to(device))
            loss = loss_function(y_hat, y.to(device))

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
        xs_neg = torch.randn(0, 2).to(device)

        ys_pos = []
        xs_pos = torch.randn(0, 2).to(device)

        model.eval()
        with torch.no_grad():
            for i, (x, y, ids) in enumerate(val_loader):
                _, y_hat = model(x.to(device))
                loss = loss_function(y_hat, y.to(device))

                # 用来计算整体指标
                total_val_loss += loss.item()
                y_hat_trans = []
                for i in range(y_hat.shape[0]):
                    y_hat_trans.append(1 if y_hat[i].item() > 0.5 else 0)
                y_hat_trans = torch.tensor(y_hat_trans)

                y_hat_total = torch.cat([y_hat_total, y_hat_trans])
                y_total = torch.cat([y_total, y])

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

        xs = torch.cat([xs_neg, xs_pos], dim=0).cpu()
        xs = xs.numpy()

        visual(xs, ys, epoch + 1)

    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 测试集验证
    y_hat_total = torch.randn(0).to(device)
    y_total = torch.randn(0).to(device)

    ys_neg = []
    xs_neg = torch.randn(0, 64).to(device)

    ys_pos = []
    xs_pos = torch.randn(0, 64).to(device)

    TP = []
    TN = []
    FP = []

    model.eval()
    with torch.no_grad():
        for i, (x, y, ids) in enumerate(test_loader):
            h, y_hat = model(x.to(device))
            loss = loss_function(y_hat, y.to(device))

            # 用来计算整体指标
            y_hat_trans = []
            for i in range(y_hat.shape[0]):
                y_hat_trans.append(1 if y_hat[i].item() > 0.5 else 0)
            y_hat_trans = torch.tensor(y_hat_trans)

            y_hat_total = torch.cat([y_hat_total, y_hat_trans])
            y_total = torch.cat([y_total, y])

            for j in range(y_hat.shape[0]):
                id = ids[j]
                fac = y[j].item()
                pre = y_hat_trans[j].item()

                temph = torch.index_select(h, dim=0, index=torch.tensor([j]).to(device))

                if fac == 0:
                    if pre == 1:
                        xs_neg = torch.cat([xs_neg, temph], dim=0)
                        ys_neg.append(0)
                        FP.append(id)

                else:

                    xs_pos = torch.cat([xs_pos, temph], dim=0)
                    ys_pos.append(1)
                if pre == 1:
                    TP.append(id)
                else:
                    TN.append(id)

    acc = accuracy_score(y_total.cpu(), y_hat_total.cpu())
    balanced_acc = balanced_accuracy_score(y_total.cpu(), y_hat_total.cpu())
    ps = precision_score(y_total.cpu(), y_hat_total.cpu())
    rc = recall_score(y_total.cpu(), y_hat_total.cpu())
    f1 = f1_score(y_total.cpu(), y_hat_total.cpu())
    c = confusion_matrix(y_total.cpu(), y_hat_total.cpu(), labels=[0, 1])

    print(f"测试集 accuracy_score: {float_to_percent(acc)}")
    print(f"测试集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
    print(f"测试集 precision_score: {float_to_percent(ps)}")
    print(f"测试集 recall_score: {float_to_percent(rc)}")
    print(f"测试集 f1_score: {float_to_percent(f1)}")
    print(f"测试集 混淆矩阵:\n {c}")

    # 对于TP（猜对了）、TN（没猜出来）、FP（猜错了） 分别取20条写进文件
    record_file = open(os.path.join('./', 'result', 'result.txt'), 'w')
    print(f"***写入文件中***\n")

    index = 0
    record_file.write("预测正确的的TN有：\n")
    for item in TP:
        record_file.write(f'    -{index}. {item}\n')
        index += 1

    index = 0
    record_file.write("实际是正样本，却被预测为负样本的TN有：\n")
    for item in TN:
        record_file.write(f'    -{index}. {item}\n')
        index += 1

    index = 0
    record_file.write("实际是负样本，被预测为正样本的FP有：\n")
    for item in FP:
        record_file.write(f'    -{index}. {item}\n')
        index += 1

    print(f"***保存tsne中***\n")
    ys = []
    ys += ys_neg
    ys += ys_pos
    ys = np.array(ys)

    xs = torch.cat([xs_neg, xs_pos], dim=0).cpu()
    xs = xs.numpy()

    visual(xs, ys, -1)
