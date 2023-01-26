import configparser
import os

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix, \
    f1_score
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from dataset import MyDataset
from util import random_unit, float_to_percent


def my_collate_fn(batch):
    ys = []
    for data in batch:
        ys.append(data.y)
    return torch.tensor(ys)


if __name__ == '__main__':
    """
    这个函数对各个项目的测试集实现了 random guess
    算法就是按照测试集里正样本比例来随机预测当前语句是不是日志语句
    然后计算结果计算四个指标
    We repeat the Random Guess 30 times for each system to reduce the biases
    重复30次对指标去个平均值吧
    """

    cf = configparser.ConfigParser()
    cf.read('config_for_eval.ini')

    project = cf.get('data', 'project')
    mode = cf.getint('mode', 'mode')

    if mode == 0:
        rate = 0.5
    else:
        rate = cf.getfloat('rate', f'{project}_rate')

    print(f"随机猜测项目{project}，以{float_to_percent(rate)}概率猜测正样本")

    root_dir = cf.get('data', 'dataDir')
    root_dir = os.path.join(root_dir, project)

    BATCH_SIZE = 64

    test_dataset = MyDataset(root=root_dir, project=project, dataset_type="test")

    if mode == 0:
        test_loader = DataLoader(dataset=test_dataset,
                                 collate_fn=my_collate_fn,
                                 sampler=ImbalancedDatasetSampler(test_dataset),
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
    else:
        test_loader = DataLoader(dataset=test_dataset,
                                 collate_fn=my_collate_fn,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

    y_hat_total = torch.randn(0)
    y_total = torch.randn(0)

    total_acc = 0.0
    total_balanced_acc = 0.0
    total_ps = 0.0
    total_rc = 0.0
    total_f1 = 0.0

    for epoch in range(30):
        print(f'第{epoch}轮随机猜测')
        for i, data in enumerate(test_loader):
            y = data

            y_hat = []
            for j in range(y.shape[0]):
                if random_unit(rate):
                    y_hat.append(1)
                else:
                    y_hat.append(0)

            y_hat = torch.tensor(y_hat)

            y_hat_total = torch.cat([y_hat_total, y_hat])
            y_total = torch.cat([y_total, y])

        acc = accuracy_score(y_total.cpu(), y_hat_total.cpu())
        balanced_acc = balanced_accuracy_score(y_total.cpu(), y_hat_total.cpu())
        ps = precision_score(y_total.cpu(), y_hat_total.cpu())
        rc = recall_score(y_total.cpu(), y_hat_total.cpu())
        f1 = f1_score(y_total.cpu(), y_hat_total.cpu())

        total_acc += acc
        total_balanced_acc += balanced_acc
        total_ps += ps
        total_rc += rc
        total_f1 += f1

    total_acc /= 30
    total_balanced_acc /= 30
    total_ps /= 30
    total_rc /= 30
    total_f1 /= 30

    print("随机猜测完成，结果为：")
    print(f"    - acc：{float_to_percent(total_acc)}")
    print(f"    - balanced_acc：{float_to_percent(total_balanced_acc)}")
    print(f"    - ps：{float_to_percent(total_ps)}")
    print(f"    - rc：{float_to_percent(total_rc)}")
    print(f"    - f1：{float_to_percent(total_f1)}")
