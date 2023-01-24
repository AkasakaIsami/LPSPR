import configparser

import torch
from torch import nn
from torch_geometric.nn import Sequential, BatchNorm, TopKPooling, Linear, global_max_pool, RGCNConv, RGATConv, GATConv


class StatementClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 读取模型配置
        cf = configparser.ConfigParser()
        cf.read('config.ini')

        self.num_layers = cf.getint('model-out', 'num_layers')
        self.dropout = cf.getfloat('model-out', 'dropout')
        self.hidden_dim = cf.getint('model-out', 'hidden_dim')
        self.input_dim = cf.getint('model-in', 'encode_dim')

        # 定义网络结构
        self.encoder = StatementEncoder()
        self.layer_0 = Sequential('x, edge_index, edge_type', [
            (RGCNConv(in_channels=self.input_dim, out_channels=self.hidden_dim, num_relations=2, is_sorted=True),
             'x, edge_index,edge_type -> x'),
            nn.Sigmoid(),
            BatchNorm(self.hidden_dim)
        ])
        self.layer_1 = Sequential('x, edge_index, edge_type', [
            (RGCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, num_relations=2, is_sorted=True),
             'x, edge_index,edge_type -> x'),
            nn.Sigmoid(),
            BatchNorm(self.hidden_dim)
        ])

        self.conv_0 = Sequential('x, edge_index, edge_type', [
            (RGATConv(self.input_dim, self.hidden_dim, num_relations=2, heads=3, dropout=self.dropout),
             'x, edge_index,edge_type -> x'),
            nn.Tanh(),
            BatchNorm(self.hidden_dim * 3)
        ])
        self.conv_1 = Sequential('x, edge_index, edge_type', [
            (RGATConv(self.hidden_dim * 3, self.hidden_dim, num_relations=2, heads=1, dropout=self.dropout),
             'x, edge_index,edge_type -> x'),
            nn.Tanh(),
            BatchNorm(self.hidden_dim)
        ])

        self.mlp = nn.Sequential(Linear(self.hidden_dim, self.hidden_dim, weight_initializer='kaiming_uniform'),
                                 nn.ReLU(),
                                 Linear(self.hidden_dim, 1, weight_initializer='kaiming_uniform'))

        # 第二个mlp 对于只有一个节点的图那就过个全连接吧- -
        self.mlp_2 = nn.Sequential(Linear(self.input_dim, self.hidden_dim, weight_initializer='kaiming_uniform'),
                                   nn.ReLU(),
                                   Linear(self.hidden_dim, 1, weight_initializer='kaiming_uniform'))

    def forward(self, astss, data):
        '''
        new_data = Data(
            index=data.index,
            edge_index=edge_index,
            edge_type=edge_type,
            y=data.y
        )
        输入一个是这样的batch
        还有一个是astss的batch
        '''

        # step1 astss先过StatementEncoder
        statements_vec = self.encoder(astss)

        # step2 过外层gnn
        x = statements_vec
        edge_index = data.edge_index
        edge_type = data.edge_type
        h = self.conv_0(x, edge_index, edge_type)
        h = self.conv_1(h, edge_index, edge_type)

        # step3 只取我们关注的batch个点
        idx = data.idx

        def idx2index(idx: torch.Tensor) -> torch.Tensor:
            """
            根据稀疏矩阵求index
            """
            index = []
            size = idx.shape[0]
            for i in range(size):
                if idx[i].item() == 1:
                    index.append(i)
            return torch.tensor(index).long().to(self.device)

        h = torch.index_select(h, dim=0, index=idx2index(idx))
        out = self.mlp(h).sigmoid()
        return out


class StatementEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 读取模型配置
        cf = configparser.ConfigParser()
        cf.read('config.ini')

        self.num_layers = cf.getint('model-in', 'num_layers')
        self.dropout = cf.getfloat('model-in', 'dropout')
        self.embedding_dim = cf.getint('embedding', 'dim')
        self.hidden_dim = cf.getint('model-in', 'hidden_dim')
        self.encode_dim = cf.getint('model-in', 'encode_dim')

        # 定义网络结构
        self.layer_0 = Sequential('x, edge_index', [
            (GATConv(in_channels=self.embedding_dim, out_channels=self.hidden_dim, heads=3,
                     dropout=self.dropout), 'x, edge_index -> x'),
            nn.Tanh(),
            BatchNorm(self.hidden_dim * 3)
        ])

        self.pooling_0 = TopKPooling(self.hidden_dim * 3, ratio=0.7)

        self.layer_1 = Sequential('x, edge_index', [
            (GATConv(in_channels=self.hidden_dim * 3, out_channels=self.hidden_dim, heads=1,
                     dropout=self.dropout), 'x, edge_index -> x'),
            nn.Tanh(),
            BatchNorm(self.hidden_dim)
        ])

        self.pooling_1 = TopKPooling(self.hidden_dim, ratio=0.5)

        self.mlp = nn.Sequential(Linear(self.hidden_dim, self.hidden_dim, weight_initializer='kaiming_uniform'),
                                 nn.ReLU(),
                                 Linear(self.hidden_dim, self.encode_dim, weight_initializer='kaiming_uniform'))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index.long()
        batch = data.batch

        h = self.layer_0(x, edge_index)
        h, edge_index, _, batch, _, _ = self.pooling_0(h, edge_index, None, batch)
        h = h.relu()
        h = self.layer_1(h, edge_index)
        h, edge_index, _, batch, _, _ = self.pooling_1(h, edge_index, None, batch)
        h = global_max_pool(h, batch)
        h = h.relu()
        out = self.mlp(h)

        return out
