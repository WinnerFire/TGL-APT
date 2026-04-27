from kairos_utils import *
from config import *

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

max_node_num = 268243  # the number of nodes in node2id table +1
min_dst_idx, max_dst_idx = 0, max_node_num
# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)


class New_GraphAttentionEmbedding(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 msg_dim,
                 time_enc,
                 num_heads=2,
                 dropout=0.1,
                 return_attention=False):
        super(New_GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        self.return_attention = return_attention

        # 第一层 GATConv
        self.gat1 = GATConv(in_channels + msg_dim + time_enc.out_channels,
                            out_channels,
                            heads=num_heads,
                            dropout=dropout,
                            concat=False)

        # 第二层 GATConv（可选）
        self.gat2 = GATConv(out_channels,
                            out_channels,
                            heads=1,
                            concat=False,
                            dropout=dropout)

    def forward(self, z, last_update, edge_index, t, msg, return_attention=None):
        """
        z: 节点当前表示 (N, in_channels)
        last_update: 节点最后更新时间 (N,)
        edge_index: 图的边 (2, E)
        t: 边的时间戳 (E,)
        msg: 边的消息 (E, msg_dim)
        return_attention: 是否返回注意力权重 (覆盖 self.return_attention)
        """

        if return_attention is None:
            return_attention = self.return_attention

        # 时间编码
        #t_enc = self.time_enc(t - last_update[edge_index[0]])
        t_diff = (t - last_update[edge_index[0]]).float()   # 或者 .to(z.dtype)
        t_enc = self.time_enc(t_diff)

        # 拼接 z, msg, time
        edge_feat = torch.cat([msg, t_enc], dim=-1)
        x = torch.cat([z, torch.zeros(z.size(0), edge_feat.size(-1), device=z.device)], dim=-1)

        # ---- GATConv 第一层 ----
        if return_attention:
            x, att1 = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index)
            att1 = None

        x = torch.relu(x)

        # ---- GATConv 第二层 ----
        if return_attention:
            x, att2 = self.gat2(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index)
            att2 = None

        # 取最后一层的注意力分布
        attention_weights = None
        if return_attention:
            if att2 is not None:
                # att2 是 (edge_index, α_ij)
                attention_weights = att2[1]
            elif att1 is not None:
                attention_weights = att1[1]

            # 聚合成节点级注意力 (简单求和归一化)
            if attention_weights is not None:
                node_attention = torch.zeros(x.size(0), device=x.device)
                edge_src = edge_index[0]
                if attention_weights.dim() == 2:  # 多头注意力
                    attention_weights = attention_weights.mean(dim=1)  

                attention_weights = attention_weights.squeeze()  # 确保是一维 [num_edges]

                node_attention = torch.zeros(z.size(0), device=z.device)  # 节点级注意力
                node_attention.scatter_add_(0, edge_src, attention_weights)

                attention_weights = node_attention / (node_attention.sum() + 1e-6)

        if return_attention:
            return x, attention_weights
        else:
            return x


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels * 8, out_channels, heads=1, concat=False,
                                     dropout=0.0, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        last_update.to(device)
        x = x.to(device)
        t = t.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)

        self.lin_seq = nn.Sequential(

            Linear(in_channels * 4, in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), out_channels)
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)
        h = self.lin_seq(h)
        return h

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
    loss=[]
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)
