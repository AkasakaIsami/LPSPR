import configparser
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from dataset import MyDataset


def visual(x, y):
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
    f.savefig('./akasaka')
    f.clear()  # 释放内存
    plt.show()


if __name__ == '__main__':
    """
    这个文件用于对初始特征进行观察
    读取zookeeper里的训练集数据观察初始特征的可视化分布
    """

    # 第一步 读取训练集
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    project = cf.get('data', 'projectName')

    root_dir = cf.get('data', 'dataDir')
    root_dir = os.path.join(root_dir, project)

    project = 'zookeeper'

    train_dataset = MyDataset(root=root_dir, project=project, dataset_type="train")
    methods_info = pd.read_pickle(os.path.join(root_dir, 'processed', 'method_info.pkl'))

    ASTOn = False
    CFGOn = True
    DFGOn = True


    def my_collate_fn(batch):
        new_datalist = []
        astss = []
        for data in batch:
            # 对于每个data 重构一个data还回去
            # 获取statemet的id id格式为clz@method@line
            # 去method_info里取出它对应函数的AST相关信息和函数相关信息

            method = data.id.split('@')[0]
            info = methods_info.loc[methods_info['id'] == method]

            if ASTOn:
                astss.extend(info['ASTs'].tolist()[0])
            else:
                for ast in info['ASTs'].tolist()[0]:
                    ast.x = torch.index_select(ast.x, dim=0, index=torch.tensor([0]))
                    ast.edge_index = torch.zeros(2, 0).long()
                astss.extend(info['ASTs'].tolist()[0])

            cfg_edge_index = info['edges'].tolist()[0][0].long()
            dfg_edge_index = info['edges'].tolist()[0][1].long()

            if (CFGOn and DFGOn) or (not CFGOn and not DFGOn):
                # 如果都关上的话 默认走两个都开
                edge_index = torch.cat([cfg_edge_index, dfg_edge_index], 1)
                len_1 = cfg_edge_index.shape[1]
                len_2 = dfg_edge_index.shape[1]
                edge_type_1 = torch.zeros(len_1, )
                edge_type_2 = torch.ones(len_2, )
                edge_type = torch.cat([edge_type_1, edge_type_2], -1).long()

                new_data = Data(
                    id=data.id,
                    idx=data.idx,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    y=data.y
                )
            elif CFGOn:
                new_data = Data(
                    id=data.id,
                    idx=data.idx,
                    edge_index=cfg_edge_index,
                    y=data.y
                )

            else:
                new_data = Data(
                    id=data.id,
                    idx=data.idx,
                    edge_index=dfg_edge_index,
                    y=data.y
                )

            new_datalist.append(new_data)

        return astss, new_datalist


    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=my_collate_fn,
                              batch_size=1,
                              shuffle=False)


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


    def eval1():
        # 实验 2.1：只看当前语句节点的AST
        ys_neg = []
        xs_neg = torch.randn(0, 128)

        ys_pos = []
        xs_pos = torch.randn(0, 128)
        for i, (asts, data) in enumerate(train_loader):
            data = data[0]
            idx = idx2index(data.idx).item()
            ast = asts[idx]
            statement_vec = ast.x.mean(axis=0)
            statement_vec = statement_vec.reshape(1, 128)

            if data.y.item() == 0:
                xs_neg = torch.cat([xs_neg, statement_vec], dim=0)
                ys_neg.append(data.y.item())
            else:
                xs_pos = torch.cat([xs_pos, statement_vec], dim=0)
                ys_pos.append(data.y.item())

        ys = []
        ys += ys_neg
        ys += ys_pos
        ys = np.array(ys)

        xs = torch.cat([xs_neg, xs_pos], dim=0)
        xs = xs.numpy()

        visual(xs, ys)


    def eval2():
        # 实验 2.2：只看当前语句节点的AST根结点
        ys_neg = []
        xs_neg = torch.randn(0, 128)

        ys_pos = []
        xs_pos = torch.randn(0, 128)

        for i, (asts, data) in enumerate(train_loader):
            data = data[0]
            idx = idx2index(data.idx).item()
            ast = asts[idx]
            statement_vec = ast.x

            if data.y.item() == 0:
                xs_neg = torch.cat([xs_neg, statement_vec], dim=0)
                ys_neg.append(data.y.item())
            else:
                xs_pos = torch.cat([xs_pos, statement_vec], dim=0)
                ys_pos.append(data.y.item())

        ys = []
        ys += ys_neg
        ys += ys_pos
        ys = np.array(ys)

        xs = torch.cat([xs_neg, xs_pos], dim=0)
        xs = xs.numpy()

        visual(xs, ys)


    def eval3():
        # 实验 2.3：考虑当前语句前的所有语句，考虑AST
        ys_neg = []
        xs_neg = torch.randn(0, 128)

        ys_pos = []
        xs_pos = torch.randn(0, 128)
        for i, (asts, data) in enumerate(train_loader):
            data = data[0]
            idx = idx2index(data.idx).item()

            statements_vec = torch.zeros(1, 128)
            for j in range(idx):
                ast = asts[j]
                statement_vec = ast.x.mean(axis=0)
                statement_vec = statement_vec.reshape(1, 128)
                statements_vec += statement_vec

            statements_vec /= idx + 1

            if data.y.item() == 0:
                xs_neg = torch.cat([xs_neg, statements_vec], dim=0)
                ys_neg.append(data.y.item())
            else:
                xs_pos = torch.cat([xs_pos, statements_vec], dim=0)
                ys_pos.append(data.y.item())

        ys = []
        ys += ys_neg
        ys += ys_pos
        ys = np.array(ys)

        xs = torch.cat([xs_neg, xs_pos], dim=0)
        xs = xs.numpy()

        visual(xs, ys)


    def eval4():
        # 实验 2.4：考虑当前语句前的所有语句，但不考虑AST
        ys_neg = []
        xs_neg = torch.randn(0, 128)

        ys_pos = []
        xs_pos = torch.randn(0, 128)

        for i, (asts, data) in enumerate(train_loader):
            data = data[0]
            idx = idx2index(data.idx).item()

            statements_vec = torch.zeros(1, 128)
            for j in range(idx):
                ast = asts[j]
                statement_vec = ast.x
                statements_vec += statement_vec

            statements_vec /= idx + 1

            if data.y.item() == 0:
                xs_neg = torch.cat([xs_neg, statements_vec], dim=0)
                ys_neg.append(data.y.item())
            else:
                xs_pos = torch.cat([xs_pos, statements_vec], dim=0)
                ys_pos.append(data.y.item())

        ys = []
        ys += ys_neg
        ys += ys_pos
        ys = np.array(ys)

        xs = torch.cat([xs_neg, xs_pos], dim=0)
        xs = xs.numpy()

        visual(xs, ys)


    eval4()
