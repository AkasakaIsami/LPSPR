import configparser
import os
import time

import pandas as pd
import pydot
import torch
from gensim.models import Word2Vec
from torch_geometric.data import InMemoryDataset, Batch, Data
from tqdm import tqdm
from typing import Tuple

from util import cut_word, float_to_percent


class MyDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, project=None, dataset_type=None, methods=None):
        self.word2vec = None
        self.embeddings = None
        self.project = project
        self.dataset_type = dataset_type
        self.methods = methods
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(MyDataset, self).__init__(root, transform, pre_transform)

        if dataset_type == "train":
            print(f"{dataset_type} using {self.processed_paths[2]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[2])

        elif dataset_type == "val":
            print(f"{dataset_type} using {self.processed_paths[3]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[3])

        elif dataset_type == "test":
            print(f"{dataset_type} using {self.processed_paths[4]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[4])

    @property
    def raw_file_names(self):
        paths = ['']
        return paths

    @property
    def processed_file_names(self):
        """
        保存一个存有函数的所有ast特征以及边的dataframe
        格式: ['类@函数名', '所有语句ast树的batch', '函数的X和edge_index']
        保存在processed/project/method_info.pkl里
        ————————————————————————————————————————————————————————————
        对于dataset 里面存了每一个语句的信息 包括
            - 语句所属函数
            - 语句在函数里的位置（第几个节点）
            - 以及自身的标签
        保存在processed/project/positive.pt 和 negative.pt里
        """

        return ['',
                'method_info.pkl',
                'train.pt',
                'val.pt',
                'test.pt']

    def process(self):
        print("数据没有被预处理过，先进行预处理。")
        start = time.time()

        # 先读取一些配置
        cf = configparser.ConfigParser()
        cf.read('config.ini')
        ratio = cf.get('data', 'ratio')
        embedding_dim = cf.getint('embedding', 'dim')

        if not os.path.exists(self.processed_paths[0]):
            os.makedirs(self.processed_paths[0])
        record_file_path = os.path.join(self.processed_paths[0], 'dataset_info.txt')
        record_file = open(record_file_path, 'w')

        # 先导入词嵌入矩阵
        project_root = self.raw_paths[0]
        word2vec_path = os.path.join(project_root, self.project + '_w2v_' + str(embedding_dim) + '.model')
        word2vec = Word2Vec.load(word2vec_path).wv
        embeddings = torch.from_numpy(word2vec.vectors).to(self.device)
        embeddings = torch.cat([embeddings, torch.zeros(1, embedding_dim).to(self.device)], dim=0)
        self.embeddings = embeddings
        self.word2vec = word2vec

        # 开始遍历所有函数
        neg_datalist = []
        pos_datalist = []
        method_info = pd.DataFrame(columns=['id', 'ASTs', 'edges', 'LOC', 'LLOC'])

        for index, item in tqdm(self.methods.iterrows(), total=len(self.methods)):
            clz = item['class']
            if clz.endswith('Test'):
                continue

            method = item['method']
            path = os.path.join(project_root, clz, method)

            files = os.listdir(path)
            method_graph_file = None
            statement_graphs_file = None
            for file in files:
                if file == '.DS_Store':
                    continue
                elif file.startswith('statements'):
                    statement_graphs_file = file
                else:
                    method_graph_file = file

            # 开始解析函数图
            method_graph_path = os.path.join(path, method_graph_file)
            method_graphs = pydot.graph_from_dot_file(method_graph_path)
            method_graph = method_graphs[0]

            y, cfg_edge_index, dfg_edge_index, lines, method_LOC, method_LLOC = self.process_method_dot(method_graph)

            # 解析所有语句图
            statements_path = os.path.join(path, statement_graphs_file)
            statement_graphs = pydot.graph_from_dot_file(statements_path)

            # 简单做个验证
            if len(statement_graphs) != len(y):
                print(f"!!!!!!!!!!!!!!!!!!{clz}的{method}解析的有问题！！！")

            asts = []
            num_statements = len(statement_graphs)
            id = clz + '_' + method
            for i in range(num_statements):
                statement_graph = statement_graphs[i]

                # Step1: 先构建存在method_info.pkl里的函数信息数据
                ast_x, ast_edge_index = self.process_statement_dot(graph=statement_graph, weight=None)
                # TODO: 保证有序 一定要一一对应
                ast_data = Data(
                    x=ast_x,
                    edge_index=ast_edge_index,
                )
                asts.append(ast_data)

                # Step2: 再构建statement数据集
                index = torch.zeros(num_statements, 1).to(self.device)
                index[i] = 1
                statement_data = Data(
                    id=id + "@" + str(lines[i]),
                    idx=index,
                    y=torch.tensor(y[i]).to(self.device)
                )

                if statement_data.y == 0:
                    neg_datalist.append(statement_data)
                else:
                    pos_datalist.append(statement_data)

            # ast_batch = Batch.from_data_list(asts)
            method_info.loc[len(method_info)] = [id, asts, (cfg_edge_index, dfg_edge_index), method_LOC,
                                                 method_LLOC]

        # 数据都在内存里了 开始保存
        end = time.time()
        print(f"全部数据读取至内存完毕，开始以{ratio}切分数据集")

        # 在切的时候，务必要注意：先放正再放负，并且要明确知道正样本有多少，这样之后采样才方便
        train_datalist = []
        val_datalist = []
        test_datalist = []

        ratios = [int(r) for r in ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * len(pos_datalist))
        val_split = train_split + int(ratios[1] / sum(ratios) * len(pos_datalist))

        train_datalist += pos_datalist[:train_split]
        val_datalist += pos_datalist[train_split:val_split]
        test_datalist += pos_datalist[val_split:]

        num_pos_in_train = len(train_datalist)
        num_pos_in_val = len(val_datalist)
        num_pos_in_test = len(test_datalist)

        train_split = int(ratios[0] / sum(ratios) * len(neg_datalist))
        val_split = train_split + int(ratios[1] / sum(ratios) * len(neg_datalist))

        train_datalist += neg_datalist[:train_split]
        val_datalist += neg_datalist[train_split:val_split]
        test_datalist += neg_datalist[val_split:]

        print(
            f"数据集切分完毕，其中训练集大小{len(train_datalist)}、验证集大小{len(val_datalist)}、测试集大小{len(test_datalist)}")

        print("现在开始对数据进行持久化存储……")
        method_info.to_pickle(self.processed_paths[1])

        data, slices = self.collate(train_datalist)
        torch.save((data, slices), self.processed_paths[2])

        data, slices = self.collate(val_datalist)
        torch.save((data, slices), self.processed_paths[3])

        data, slices = self.collate(test_datalist)
        torch.save((data, slices), self.processed_paths[4])

        '''
        每次构建数据集时，返回当前数据集的信息。信息包括：
            -  数据总量
            -  正样本量
            -  负样本量
            -  总函数量
            -  耗时
            还有训练集、验证集、测试集的详细信息。信息包括：
            -   集合数据总量
            -   集合数据正样本量
            -   集合数据负样本量
        '''
        record_file.write(f"数据集构建完成，下面是一些数据集相关信息：\n")
        record_file.write(f"    - 目标项目：{self.project}\n")
        record_file.write(f"    - 数据总量：{len(pos_datalist) + len(neg_datalist)}\n")
        record_file.write(f"    - 正样本量：{len(pos_datalist)}\n")
        record_file.write(f"    - 负样本量：{len(neg_datalist)}\n")
        temp = float_to_percent(len(pos_datalist) / (len(pos_datalist) + len(neg_datalist)))
        record_file.write(f"    - 正样本含量：{temp}\n")
        record_file.write(f"    - 总函数量：{len(method_info)}\n")
        record_file.write(f"    - 总耗时：{end - start}秒\n")
        record_file.write(f"下面是一些关于训练集、验证集、测试集的相关信息：(分层采样，每个数据集里正样本含量相同)\n")
        record_file.write(f"    - 训练集：\n")
        record_file.write(f"        * 数据总量：{len(train_datalist)}\n")
        record_file.write(f"        * 正样本量：{num_pos_in_train}\n")
        record_file.write(f"        * 负样本量：{len(train_datalist) - num_pos_in_train}\n")
        record_file.write(f"    - 验证集：\n")
        record_file.write(f"        * 数据总量：{len(val_datalist)}\n")
        record_file.write(f"        * 正样本量：{num_pos_in_val}\n")
        record_file.write(f"        * 负样本量：{len(val_datalist) - num_pos_in_val}\n")
        record_file.write(f"    - 测试集：\n")
        record_file.write(f"        * 数据总量：{len(test_datalist)}\n")
        record_file.write(f"        * 正样本量：{num_pos_in_test}\n")
        record_file.write(f"        * 负样本量：{len(test_datalist) - num_pos_in_test}")
        record_file.close()

    def process_method_dot(self, graph) -> Tuple[list, torch.Tensor, torch.Tensor, list, int, int]:
        """
        处理函数的dot，返回当前函数的图结构
        返回值：y, cfg_edge_index, dfg_edge_index, lines, method_LOC, method_LLOC
        """
        nodes = graph.get_node_list()
        if len(graph.get_node_list()) > 0 and graph.get_node_list()[-1].get_name() == '"\\n"':
            nodes = graph.get_node_list()[:-1]
        node_num = len(nodes)

        tempLOC = 0
        tempLLOC = 0
        y = []

        # 存了每个语句的行数 数量和节点数量对应
        lines = []

        for i in range(node_num):
            node = nodes[i]
            line = node.get_attributes()['line']
            lines.append(int(line))

            tempLOC = tempLOC + 1
            if 'true' in node.get_attributes()['isLogged']:
                tempLLOC = tempLLOC + 1
                y.append(1)
            else:
                y.append(0)

        edges = graph.get_edge_list()
        edge_0_cfg = []
        edge_1_cfg = []
        edge_0_dfg = []
        edge_1_dfg = []

        for edge in edges:
            source = int(edge.get_source()[1:])
            destination = int(edge.get_destination()[1:])
            color = edge.get_attributes()['color']

            if color == 'red':
                edge_0_cfg.append(source)
                edge_1_cfg.append(destination)
            elif color == 'green':
                edge_0_dfg.append(source)
                edge_1_dfg.append(destination)

        edge_0_cfg = torch.as_tensor(edge_0_cfg).to(self.device)
        edge_1_cfg = torch.as_tensor(edge_1_cfg).to(self.device)
        edge_0_cfg = edge_0_cfg.reshape(1, len(edge_0_cfg))
        edge_1_cfg = edge_1_cfg.reshape(1, len(edge_1_cfg))

        edge_0_dfg = torch.as_tensor(edge_0_dfg).to(self.device)
        edge_1_dfg = torch.as_tensor(edge_1_dfg).to(self.device)
        edge_0_dfg = edge_0_dfg.reshape(1, len(edge_0_dfg))
        edge_1_dfg = edge_1_dfg.reshape(1, len(edge_1_dfg))

        cfg_edge_index = torch.cat([edge_0_cfg, edge_1_cfg], dim=0)
        dfg_edge_index = torch.cat([edge_0_dfg, edge_1_dfg], dim=0)

        return y, cfg_edge_index, dfg_edge_index, lines, tempLOC, tempLLOC

    def process_statement_dot(self, graph, weight):
        """
        这个函数返回ST-AST的特征矩阵和邻接矩阵
        特征矩阵需要根据语料库构建……

        :param weight:
        :param graph: ST-AST
        :return: 特征矩阵和邻接矩阵
        """

        def word_to_vec(token):
            """
            词转词嵌入
            :param token:
            :return: 返回一个代表词嵌入的ndarray
            """
            max_token = self.word2vec.vectors.shape[0]
            index = [self.word2vec.key_to_index[token] if token in self.word2vec.key_to_index else max_token]
            return self.embeddings[index].to(self.device)

        def tokens_to_embedding(tokens, weight):
            """
            对于多token组合的节点 可以有多种加权求和方式
            这里简单的求平均先

            :param tokens:节点的token序列
            :return: 最终的节点向量
            """
            result = torch.zeros([1, 128], dtype=torch.float).to(self.device)

            for token in tokens:
                token_embedding = word_to_vec(token)
                if weight is not None:
                    token_weight = weight[token] if weight.has_key(token) else 0
                    token_embedding = token_embedding * token_weight
                result = result + token_embedding

            count = len(tokens)
            result = result / count
            return result

        x = []
        nodes = graph.get_node_list()
        if len(graph.get_node_list()) > 0 and graph.get_node_list()[-1].get_name() == '"\\n"':
            nodes = graph.get_node_list()[:-1]

        for node in nodes:
            node_str = node.get_attributes()['label']
            # token 可能是多种形势，要先切分
            tokens = cut_word(node_str)
            # 多token可以考虑不同的合并方式
            node_embedding = tokens_to_embedding(tokens, weight)
            x.append(node_embedding)

        x = torch.cat(x).to(self.device)

        edges = graph.get_edge_list()
        edge_0 = []
        edge_1 = []

        for edge in edges:
            source = int(edge.get_source()[1:])
            destination = int(edge.get_destination()[1:])
            edge_0.append(source)
            edge_1.append(destination)

        edge_0 = torch.as_tensor(edge_0, dtype=torch.int).to(self.device)
        edge_1 = torch.as_tensor(edge_1, dtype=torch.int).to(self.device)
        edge_0 = edge_0.reshape(1, len(edge_0))
        edge_1 = edge_1.reshape(1, len(edge_1))

        edge_index = torch.cat([edge_0, edge_1], dim=0)

        return x, edge_index
