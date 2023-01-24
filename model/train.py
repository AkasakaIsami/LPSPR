import configparser
import os
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torchsampler import ImbalancedDatasetSampler

from dataset import MyDataset


def train(train_dataset: MyDataset, val_dataset: MyDataset, methods_info: pd.DataFrame, result_dir: str):
    # 读取超参配置
    # 模型配置到模型里去读
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    # 读取超参配置
    EPOCHS = cf.getint('train', 'epoch')
    BATCH_SIZE = cf.getint('train', 'batchSize')

    LR = cf.getfloat('train', 'learningRate')
    ALPHA = cf.getfloat('train', 'alpha')
    GAMMA = cf.getfloat('train', 'gamma')

    # 读取特征选择
    ASTOn = cf.get('eval-config', 'ASTOn')
    CFGOn = cf.get('eval-config', 'CFGOn')
    DFGOn = cf.get('eval-config', 'DFGOn')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取采样配置

    # 在正式开始训练前，先设置一下日志持久化的配置
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    start_time = datetime.now()
    start_time_str = datetime.strftime(start_time, '%Y-%m-%d_%H:%M:%S')
    record_file_name = start_time_str + '_train_info_' + '.txt'

    # 训练前先定义一下batch的读取方式
    def my_collate_fn(batch):
        new_datalist = []
        astss = []
        for data in batch:
            # 对于每个data 重构一个data还回去
            # 获取statemet的id id格式为clz@method@line
            # 去method_info里取出它对应函数的AST相关信息和函数相关信息

            method = data.id.split('@')[0]
            # TODO: 用来最后算TN和FP的
            line = data.id.split('@')[1]

            info = pd.DataFrame(methods_info['id'] == method)

            # TODO: 每个AST只取第一行
            astss += info['ASTs']

            cfg_edge_index = info['edges'][0]
            dfg_edge_index = info['edges'][1]

            if CFGOn and DFGOn:
                edge_index = torch.cat([cfg_edge_index, dfg_edge_index], 1).long()
                len_1 = cfg_edge_index.shape[1]
                len_2 = dfg_edge_index.shape[1]
                edge_type_1 = torch.zeros(len_1, )
                edge_type_2 = torch.ones(len_2, )
                edge_type = torch.cat([edge_type_1, edge_type_2], -1).long()

                new_data = Data(
                    index=data.index,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    y=data.y
                )
            elif CFGOn:
                new_data = Data(
                    index=data.index,
                    edge_index=info['edges'][0],
                    y=data.y
                )

            else:
                new_data = Data(
                    index=data.index,
                    edge_index=info['edges'][1],
                    y=data.y
                )

            new_datalist.append(new_data)

        return Batch.from_data_list(astss), Batch.from_data_list(new_datalist)

    # 正式开始训练！
    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=my_collate_fn,
                              sampler=ImbalancedDatasetSampler(train_dataset),
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    val_loader = DataLoader(dataset=val_dataset,
                            collate_fn=my_collate_fn,
                            batch_size=BATCH_SIZE,
                            shuffle=False)
