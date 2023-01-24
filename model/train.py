import configparser
import os
from datetime import datetime
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torchinfo import summary
from torchsampler import ImbalancedDatasetSampler

from dataset import MyDataset
from model import StatementClassifier
from util import float_to_percent


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
    record_file = open(os.path.join(result_dir, record_file_name), 'w')

    record_file.write(f"本次实验开始时间：{start_time_str}\n")
    record_file.write(f"实验配置如下：\n")
    record_file.write(f"    - 是否使用AST：{ASTOn}\n")
    record_file.write(f"    - 是否使用CFG：{CFGOn}\n")
    record_file.write(f"    - 是否使用DFG：{DFGOn}\n")
    record_file.write(f"模型配置如下：\n")
    record_file.write(f"    - EPOCHS：{EPOCHS}\n")
    record_file.write(f"    - BATCH_SIZE：{BATCH_SIZE}\n")
    record_file.write(f"    - LEARNING_RATE：{LR}\n")

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

            info = methods_info.loc[methods_info['id'] == method]

            # TODO: 每个AST只取第一行
            astss += info['ASTs'].tolist()[0]

            cfg_edge_index = info['edges'].tolist()[0][0]
            dfg_edge_index = info['edges'].tolist()[0][1]

            if CFGOn and DFGOn:
                edge_index = torch.cat([cfg_edge_index, dfg_edge_index], 1).long()
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
                    edge_index=info['edges'][0],
                    y=data.y
                )

            else:
                new_data = Data(
                    id=data.id,
                    idx=data.idx,
                    edge_index=info['edges'][1],
                    y=data.y
                )

            new_datalist.append(new_data)

        return Batch.from_data_list(astss), Batch.from_data_list(new_datalist)

    # 准备开始训练
    # 先定义数据读取方式 过采样+欠采样
    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=my_collate_fn,
                              sampler=ImbalancedDatasetSampler(train_dataset),
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    val_loader = DataLoader(dataset=val_dataset,
                            collate_fn=my_collate_fn,
                            sampler=ImbalancedDatasetSampler(val_dataset),
                            batch_size=BATCH_SIZE,
                            shuffle=False)

    # 定义模型相关的东西
    model = StatementClassifier().to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_function = torch.nn.MSELoss().to(device)

    record_file.write(f"模型结构如下：\n")
    record_file.write(str(summary(model)) + '\n')

    # 定义用于评估预测结果的东西
    best_acc = 0.0
    best_model = model

    # 定义控制日志打印的东西
    total_train_step = 0
    start = time.time()

    # 正式开始训练！
    for epoch in range(EPOCHS):
        print(f'------------第 {epoch + 1} 轮训练开始------------')
        record_file.write(f'------------第 {epoch + 1} 轮训练开始------------\n')

        model.train()
        for i, (astss, data) in enumerate(train_loader):
            y_hat = model(astss.to(device), data.to(device))
            y_hat = y_hat.reshape(y_hat.shape[0], )
            y = data.y.float().to(device)
            loss = loss_function(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print(f"训练次数: {total_train_step}, Loss: {loss.item()}")
                record_file.write(f"训练次数: {total_train_step}, Loss: {loss.item()}\n")

        # 开始验证！
        total_val_loss = 0.0
        y_hat_total = torch.randn(0).to(device)
        y_total = torch.randn(0).to(device)

        model.eval()
        with torch.no_grad():
            for i, (astss, data) in enumerate(val_loader):
                y_hat = model(astss.to(device), data.to(device))
                y_hat = y_hat.reshape(y_hat.shape[0], )
                y = data.y.to(device)
                loss = loss_function(y_hat, y)

                # 用来计算验证集整体指标
                total_val_loss += loss.item()
                y_hat_total = torch.cat([y_hat_total, y_hat])
                y_total = torch.cat([y_total, y])

        for i in range(y_hat_total.shape[0]):
            y_hat_total[i] = 1 if y_hat_total[i] >= 0.5 else 0

        acc = accuracy_score(y_total.cpu(), y_hat_total.cpu())
        balanced_acc = balanced_accuracy_score(y_total.cpu(), y_hat_total.cpu())
        ps = precision_score(y_total.cpu(), y_hat_total.cpu())
        rc = recall_score(y_total.cpu(), y_hat_total.cpu())
        f1 = f1_score(y_total.cpu(), y_hat_total.cpu())
        c = confusion_matrix(y_total.cpu(), y_hat_total.cpu(), labels=[0, 1])

        print(f"验证集整体Loss: {total_val_loss}")
        print(f"验证集 accuracy_score: {float_to_percent(acc)}")
        print(f"验证集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
        print(f"验证集 precision_score: {float_to_percent(ps)}")
        print(f"验证集 recall_score: {float_to_percent(rc)}")
        print(f"验证集 f1_score: {float_to_percent(f1)}")
        print(f"验证集 混淆矩阵:\n {c}")

        record_file.write(f"验证集整体Loss: {total_val_loss}\n")
        record_file.write(f"验证集 accuracy_score: {float_to_percent(acc)}\n")
        record_file.write(f"验证集 balanced_accuracy_score: {float_to_percent(balanced_acc)}\n")
        record_file.write(f"验证集 precision_score: {float_to_percent(ps)}\n")
        record_file.write(f"验证集 recall_score: {float_to_percent(rc)}\n")
        record_file.write(f"验证集 f1_score: {float_to_percent(f1)}\n")
        record_file.write(f"验证集 混淆矩阵:\n {c}\n")

        # 主要看balanced_accuracy_score
        if balanced_acc > best_acc:
            best_model = model
            best_acc = balanced_acc

    end = time.time()
    print(f"训练完成，共耗时{end - start}秒。最佳balanced accuracy是{float_to_percent(best_acc)}。现在开始保存数据...")
    record_file.write(f"训练完成，共耗时{end - start}秒。最佳balanced accuracy是{best_acc}\n")

    # 保存模型
    model_file_name = start_time_str + '_model@' + float_to_percent(best_acc) + '.pth'
    model_save_path = os.path.join(result_dir, model_file_name)
    torch.save(best_model, model_save_path)
    print('模型保存成功！')

    record_file.write(
        f"——————————只有看到这条语句，并且对应的模型文件也成功保存了，这个日志文件的内容才有效！（不然就是中断了）——————————\n")
    record_file.close()
    return best_model, os.path.join(result_dir, record_file_name)
