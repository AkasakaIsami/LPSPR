import configparser
import os

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from util import float_to_percent


def test(model, test_dataset, methods_info, record_file_path: str):
    record_file = open(os.path.join(record_file_path), 'a')
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    BATCH_SIZE = cf.getint('train', 'batchSize')

    # 读取特征选择
    ASTOn = cf.get('eval-config', 'ASTOn')
    CFGOn = cf.get('eval-config', 'CFGOn')
    DFGOn = cf.get('eval-config', 'DFGOn')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        return Batch.from_data_list(astss).to(device), Batch.from_data_list(new_datalist).to(device)

    test_loader = DataLoader(dataset=test_dataset,
                             collate_fn=my_collate_fn,
                             sampler=ImbalancedDatasetSampler(test_dataset),
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    y_hat_total = torch.randn(0).to(device)
    y_total = torch.randn(0).to(device)

    model.eval()
    with torch.no_grad():
        for i, (astss, data) in enumerate(test_loader):
            y_hat = model(astss, data)
            y_hat = y_hat.reshape(y_hat.shape[0], )
            y = data.y

            # 用来计算测试集整体指标
            y_hat_total = torch.cat([y_hat_total, y_hat])
            y_total = torch.cat([y_total, y])

    for i in range(y_hat_total.shape[0]):
        y_hat_total[i] = 1 if y_hat_total[i] >= 0.5 else 0

    acc = accuracy_score(y_total, y_hat_total)
    balanced_acc = balanced_accuracy_score(y_total, y_hat_total)
    ps = precision_score(y_total, y_hat_total)
    rc = recall_score(y_total, y_hat_total)
    f1 = f1_score(y_total, y_hat_total)
    c = confusion_matrix(y_total, y_hat_total, labels=[0, 1])

    print(f"测试集 accuracy_score: {float_to_percent(acc)}")
    print(f"测试集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
    print(f"测试集 precision_score: {float_to_percent(ps)}")
    print(f"测试集 recall_score: {float_to_percent(rc)}")
    print(f"测试集 f1_score: {float_to_percent(f1)}")
    print(f"测试集 混淆矩阵:\n {c}")

    record_file.write("下面是测试集结果：\n")
    record_file.write(f"测试集 accuracy_score: {float_to_percent(acc)}\n")
    record_file.write(f"测试集 balanced_accuracy_score: {float_to_percent(balanced_acc)}\n")
    record_file.write(f"测试集 precision_score: {float_to_percent(ps)}\n")
    record_file.write(f"测试集 recall_score: {float_to_percent(rc)}\n")
    record_file.write(f"测试集 f1_score: {float_to_percent(f1)}\n")
    record_file.write(f"测试集 混淆矩阵:\n {c}")

    record_file.close()
