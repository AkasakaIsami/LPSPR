import configparser
import os

import pandas as pd

from dataset import MyDataset
from test import test
from train import train


def dictionary_and_embedding(raw_dir, project):
    embedding_dim = cf.getint('embedding', 'dim')

    corpus_file_path = os.path.join(raw_dir,  project + '_corpus.txt')
    model_file_name = project + "_w2v_" + str(embedding_dim) + '.model'

    save_path = os.path.join(raw_dir, model_file_name)
    if os.path.exists(save_path):
        return

    from gensim.models import word2vec

    corpus = word2vec.LineSentence(corpus_file_path)
    w2v = word2vec.Word2Vec(corpus, vector_size=embedding_dim, workers=16, sg=1, min_count=3)
    w2v.save(save_path)


def get_data(raw_dir: str, process_dir: str) -> pd.DataFrame:
    """
    根据项目目录下的文件结构来收集所有的函数
    """
    if os.path.exists(process_dir):
        return None

    all_methods = pd.DataFrame(columns=['class', 'method'])

    project_dir = os.path.join(raw_dir)
    classes = os.listdir(project_dir)

    for clz in classes:
        method_dir = os.path.join(project_dir, clz)
        if os.path.isfile(method_dir):
            continue

        if clz.endswith('Test'):
            continue

        methods = os.listdir(method_dir)
        for method in methods:
            if method == ".DS_Store":
                continue
            all_methods.loc[len(all_methods)] = [clz, method]

    all_methods = all_methods.sample(frac=1)
    return all_methods


def make_dataset(methods: pd.DataFrame, root_dir: str, project: str):
    """
    根据函数列表制作数据集
    已经切分完了
    均衡采样的事情交给dataloader去做
    """
    train_dataset = MyDataset(root=root_dir, project=project, dataset_type="train", methods=methods)
    val_dataset = MyDataset(root=root_dir, project=project, dataset_type="val")
    test_dataset = MyDataset(root=root_dir, project=project, dataset_type="test")
    methods_info = pd.read_pickle(os.path.join(root_dir, 'processed', 'method_info.pkl'))

    # 读配置 开始切分数据集
    return train_dataset, val_dataset, test_dataset, methods_info


if __name__ == '__main__':
    # 读取配置
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    project = cf.get('data', 'projectName')

    root_dir = cf.get('data', 'dataDir')
    root_dir = os.path.join(root_dir, project)
    raw_dir = os.path.join(root_dir, 'raw')
    process_dir = os.path.join(root_dir, 'processed')

    print(f'开始数据预处理（目标项目为{project}）...')
    print('step1: 词嵌入训练...')
    dictionary_and_embedding(raw_dir, project)

    print('step2: 获取源数据...')
    method_list = get_data(raw_dir, process_dir)

    print('step3: 制作数据集...')
    train_dataset, val_dataset, test_dataset, methods_info = make_dataset(method_list, root_dir, project)

    print('step4: 开始训练...')
    result_dir = os.path.join(root_dir, 'result')
    # model, record_file_path = train(train_dataset, val_dataset, methods_info, result_dir)

    print('step5: 开始测试...')
    # test(model, test_dataset, methods_info, record_file_path)

    print('完成...')
