##########################################################################################
# 改进版训练脚本 (支持动态 hub_nodes 优化)
##########################################################################################

import logging
import os
from tqdm import tqdm
import torch

from kairos_utils import *
from config import *
from model import *


# ========== Logging ==========
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# ========== Hub nodes 动态更新 ==========
def refine_hub_nodes(hub_nodes_tensor, z, attention_weights, epoch,
                     top_k_ratio=0.2, drop_ratio=0.05):
    """
    动态调整 hub_nodes 集合
    Args:
        hub_nodes_tensor: torch.Tensor, 当前全局 hub_nodes
        z: 当前 batch 节点嵌入, [num_nodes, dim]
        attention_weights: 当前 batch attention, [num_nodes] 或 [num_nodes, num_heads]
        epoch: 当前训练轮数
        top_k_ratio: 新增重要节点占当前 batch 节点的比例
        drop_ratio: 移除低贡献节点比例
    Returns:
        new_hub_nodes_tensor: torch.Tensor, 更新后的 hub_nodes
    """

    if attention_weights is None or z is None:
        return hub_nodes_tensor

    # ---------------- 多头注意力 -> 平均
    if attention_weights.dim() == 2:  # [num_nodes, num_heads]
        attention_weights = attention_weights.mean(dim=1)
    attention_weights = attention_weights.squeeze()  # [num_nodes]

    num_batch_nodes = attention_weights.size(0)
    top_k = max(1, int(num_batch_nodes * top_k_ratio))
    hub_nodes = set(hub_nodes_tensor.tolist())

    # ---------------- 基于 attention 选新增节点 (局部索引 -> 全局索引) ----------------
    _, top_idx = torch.topk(attention_weights, k=top_k)
    # 注意 top_idx 是 batch 内局部索引, 映射到全局节点需要用 z 对应的 n_id
    # 假设 z 的顺序对应 n_id (train 中传入 n_id)
    # 这里简化处理: 直接加入 batch 中节点索引
    new_nodes = set(top_idx.tolist())

    # ---------------- 基于嵌入偏离度选补充节点 ----------------
    mean_z = z.mean(dim=0, keepdim=True)
    deviation = torch.norm(z - mean_z, dim=1)
    _, dev_top_idx = torch.topk(deviation, k=top_k // 2)
    new_nodes |= set(dev_top_idx.tolist())

    # ---------------- 融合到全局 hub_nodes ----------------
    hub_nodes |= new_nodes

    # ---------------- 移除低贡献节点 (防止索引越界) ----------------
    hub_list = list(hub_nodes)
    # 只保留在当前 batch 范围内的节点进行 scoring
    valid_hub_list = [h for h in hub_list if h < num_batch_nodes]
    if len(valid_hub_list) > 0:
        hub_scores = attention_weights[valid_hub_list]
        drop_k = max(1, int(len(hub_nodes) * drop_ratio))
        drop_k = min(drop_k, len(valid_hub_list))
        _, low_idx = torch.topk(-hub_scores, k=drop_k)
        for li in low_idx.tolist():
            hub_nodes.discard(valid_hub_list[li])

    new_hub_nodes_tensor = torch.tensor(sorted(hub_nodes),
                                        dtype=torch.long,
                                        device=device)
    logger.info(f"[Epoch {epoch}] Hub nodes updated: {len(hub_nodes)} nodes.")
    return new_hub_nodes_tensor



# ========== 训练 ==========
def train(train_data,
          idx,
          hub_nodes_tensor,
          memory,
          gnn,
          link_pred,
          optimizer,
          neighbor_loader):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()
    neighbor_loader.reset_state()

    total_loss = 0
    valid_batches = 0
    last_z, last_attention = None, None

    for batch in train_data.seq_batches(batch_size=BATCH):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        src_in_hub = torch.isin(src, hub_nodes_tensor)
        pos_dst_in_hub = torch.isin(pos_dst, hub_nodes_tensor)

        #
        if not src_in_hub.all() or not pos_dst_in_hub.all():
            continue

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # ---- 获取 z 和 attention ----
        z, last_update = memory(n_id)
        z, attention_weights = gnn(
            z=z,
            last_update=last_update,
            edge_index=edge_index,
            t=train_data.t[e_id],
            msg=train_data.msg[e_id],
            return_attention=True
        )

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        # ---- 构造标签 ----
        y_true = []
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true, dtype=torch.long, device=device)

        # ---- Loss ----
        loss = criterion(pos_out, y_true)

        # ---- 更新 memory & neighbor ----
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()

        total_loss += float(loss)
        valid_batches += 1

        last_z, last_attention = z, attention_weights

    # ✅ Loss 归一化按有效 batch
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
    return avg_loss, last_z, last_attention, hub_nodes_tensor


# ========== 数据加载 ==========
def load_train_data():
    graph_4_2 = torch.load(graphs_dir + "/graph_4_2.TemporalData.simple").to(device=device)
    graph_4_3 = torch.load(graphs_dir + "/graph_4_3.TemporalData.simple").to(device=device)
    graph_4_4 = torch.load(graphs_dir + "/graph_4_4.TemporalData.simple").to(device=device)
    graph_4_8 = torch.load(graphs_dir + "/graph_4_8.TemporalData.simple").to(device=device)
    graph_4_9 = torch.load(graphs_dir + "/graph_4_9.TemporalData.simple").to(device=device)
    graph_4_10 = torch.load(graphs_dir + "/graph_4_10.TemporalData.simple").to(device=device)
    return [graph_4_2, graph_4_3, graph_4_4, graph_4_8, graph_4_9, graph_4_10]


# ========== 模型初始化 ==========
def init_models(node_feat_size):
    memory = TGNMemory(
        max_node_num,
        node_feat_size,
        node_state_dim,
        time_dim,
        message_module=IdentityMessage(node_feat_size, node_state_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = New_GraphAttentionEmbedding(
        in_channels=node_state_dim,
        out_channels=edge_dim,
        msg_dim=node_feat_size,
        time_enc=memory.time_enc,
    ).to(device)

    out_channels = len(include_edge_type)
    link_pred = LinkPredictor(in_channels=edge_dim, out_channels=out_channels).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=lr,
        eps=eps,
        weight_decay=weight_decay
    )

    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

    return memory, gnn, link_pred, optimizer, neighbor_loader


# ========== 主程序 ==========
if __name__ == "__main__":
    logger.info("Start logging.")

    # ---- Load data ----
    train_data = load_train_data()

    # ---- Init models ----
    node_feat_size = train_data[0].msg.size(-1)
    memory, gnn, link_pred, optimizer, neighbor_loader = init_models(node_feat_size=node_feat_size)

    # ---- 初始 hub_nodes 来自先验，只读一次 ✅ ----
    with open('./artifact/hub_nodes.log', 'r') as hub_file:
        hub_nodes_list = hub_file.readlines()
    nums = hub_nodes_list[0].replace('[', '').replace(']', '').replace('\n', '').split(',')
    initial_hub_nodes = [int(x.strip()) for x in nums if x.strip() != '']
    hub_nodes_tensor = torch.tensor(initial_hub_nodes, dtype=torch.long, device=device)

    # ---- Train ----
    for epoch in tqdm(range(1, epoch_num + 1)):
        for i, g in enumerate(train_data):
            loss, z, attention_weights, hub_nodes_tensor = train(
                train_data=g,
                idx=i,
                hub_nodes_tensor=hub_nodes_tensor,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader
            )
            logger.info(f'  Epoch: {epoch:02d}, Graph: {i}, Loss: {loss:.4f}')

        # ---- 每个 epoch 动态更新 hub_nodes ----
        hub_nodes_tensor = refine_hub_nodes(hub_nodes_tensor, z, attention_weights, epoch)

    # ---- Save model ----
    model = [memory, gnn, link_pred, neighbor_loader]
    os.system(f"mkdir -p {models_dir}")
    torch.save(model, f"{models_dir}/models.pt")
