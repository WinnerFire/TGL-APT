def get_entity_list(data_lines):
    """
    data_lines: ["src dst type", ...]
    返回所有出现过的节点编号（int），去重后按升序排列
    """
    entity_set = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            entity_set.append(int(parts[0]))
            entity_set.append(int(parts[1]))
    return entity_set

def get_relation_list(data_lines):
    """
    data_lines: ["src dst type", ...]
    返回: [[src(int), dst(int), type(str)], ...]
    """
    relation_list = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) == 3:
            src = int(parts[0])
            dst = int(parts[1])
            typ = parts[2]
            relation_list.append([src, dst, typ])
    return relation_list

def get_indegree(entity_list,relation_list):
    indegree_list = []
    for i1 in range(len(entity_list)):
        indegree_list.append({entity_list[i1]:0})
    for i in range(len(relation_list)):
        number = int(relation_list[i][1])
        if number == 0 and relation_list[i][2] == "RF":
            continue
        else:
            indegree_list[number][entity_list[number]] += 1
    return indegree_list

def get_outdegree(entity_list,relation_list):
    outdegree_list = []
    for i1 in range(len(entity_list)):
        outdegree_list.append({entity_list[i1]:0})
    for i in range(len(relation_list)):
        number = int(relation_list[i][0])
        outdegree_list[number][entity_list[number]] += 1
    return outdegree_list

def get_indegree_zero(indegree_list):
    indegree_zero = []
    count = 0
    for dic in indegree_list:
        value = dic.values()
        # print(value)
        if 0 in list(value):
            indegree_zero.append(count)
        count += 1
    return indegree_zero

def get_outdegree_zero(outdegree_list):
    outdegree_zero = []
    count1 = 0
    for dic_1 in outdegree_list:
        value = dic_1.values()
        # print(value)
        if 0 in list(value):
            outdegree_zero.append(count1)
        count1 += 1
    return outdegree_zero

def construct_Adjacency_Matrix_Connected(entity_list,relation_list):
    #bond2num = {"RD": 0, "WR": 1, "EX": 2, "UK": 3, "CD": 4, "FR": 5, "IJ": 6, "ST": 7, "RF": 8}
    #entity_list = get_entity_list(data)
    #relation_list = get_relation_list(data)
    #node_to_index = {node: index for index, node in enumerate(entity_list)}
    num_nodes = len(entity_list)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in relation_list:
        # source, target = edge
        source_idx = int(edge[0])
        target_idx = int(edge[1])
        adj_matrix[source_idx, target_idx] = 1
    return adj_matrix

def construct_Adjacency_Matrix_Value(entity_list,relation_list):
    bond2num = {'EVENT_WRITE': 1,'EVENT_READ': 2,'EVENT_CLOSE': 3,'EVENT_OPEN': 4,'EVENT_EXECUTE': 5,'EVENT_SENDTO': 6,'EVENT_RECVFROM': 7}
    num_nodes = len(entity_list)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in relation_list:
        source_idx = int(edge[0])
        target_idx = int(edge[1])
        edge_idx = bond2num[edge[2]]+1
        adj_matrix[source_idx, target_idx] = edge_idx
    return adj_matrix

def construct_Adjacency_Matrix_Timestamp(entity_list,relation_list):
    num_nodes = len(entity_list)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in relation_list:
        source_idx = int(edge[0])
        target_idx = int(edge[1])
        adj_matrix[source_idx, target_idx] = int(edge[3])
    return adj_matrix

def find_special_node(adj_timestamp,entity_list):
    special_node = []
    for node1 in range(len(entity_list)):
        out_edge = []
        in_edge = []
        for node2 in range(len(entity_list)):
            if adj_timestamp[node1][node2] != 0:
                out_edge.append(adj_timestamp[node1][node2])
            if adj_timestamp[node2][node1] != 0:
                in_edge.append(adj_timestamp[node2][node1])
        all_less = has_smaller_element(out_edge,in_edge)
        if all_less:
            if (len(out_edge)*len(in_edge) != 0):
                special_node.append(node1)
    return special_node

def find_special_start_node(adj_timestamp,entity_list):
    special_node = []
    for node1 in range(len(entity_list)):
        out_edge = []
        in_edge = []
        for node2 in range(len(entity_list)):
            if adj_timestamp[node1][node2] != 0:
                out_edge.append(adj_timestamp[node1][node2])
            if adj_timestamp[node2][node1] != 0:
                in_edge.append(adj_timestamp[node2][node1])
        all_less = one_out_edges_smaller_than_all_in_edges(out_edge, in_edge)
        if all_less:
            if (len(out_edge) * len(in_edge) != 0):
                special_node.append(node1)
    return special_node

def find_special_end_node(adj_timestamp,entity_list):
    special_node = []
    for node1 in range(len(entity_list)):
        out_edge = []
        in_edge = []
        for node2 in range(len(entity_list)):
            if adj_timestamp[node1][node2] != 0:
                out_edge.append(adj_timestamp[node1][node2])
            if adj_timestamp[node2][node1] != 0:
                in_edge.append(adj_timestamp[node2][node1])
        all_less = one_in_edges_bigger_than_all_out_edges(out_edge, in_edge)
        if all_less:
            if (len(out_edge) * len(in_edge) != 0):
                special_node.append(node1)
    return special_node

def find_all_flows_between_two(graph, start_node, end_node, path, paths):
    path.append(start_node)
    if start_node == end_node:
        paths.append(list(path))
    else:
        for neighbor, connected in enumerate(graph[start_node]):
            if connected and neighbor not in path:
                find_all_flows_between_two(graph, neighbor, end_node, path, paths)
    path.pop()

def find_all_flows(indegree_zero_plus,outdegree_zero_plus,adj_matrix_connected):
    all_paths_find = []
    for start_node in indegree_zero_plus:
        for end_node in outdegree_zero_plus:
            all_paths = []
            find_all_flows_between_two(adj_matrix_connected, start_node, end_node, [], all_paths)
            if all_paths:
                for path in all_paths:
                    all_paths_find.append(path)
    return all_paths_find


window_size_ns = 15 * 60 * int(1e9)  # 15分钟窗口，纳秒
if len(events) == 0:
    return

start_time = events[0][5]
end_time = events[-1][5]
current_window_start = start_time
current_window_end = current_window_start + window_size_ns

window_events = []
window_idx = 0

for e in events:
    ts = e[5]
    if ts < current_window_end:
        window_events.append(e)
    else:
        # 处理当前窗口
        if window_events:
            # 构造 entity_list 和 relation_list
            data_lines = []
            for ev in window_events:
                # 构造格式："{src} {dst} {type}"
                data_lines.append(f"{ev[1]} {ev[4]} {ev[2]}")
            entity_list = get_entity_list(data_lines)
            relation_list = get_relation_list(data_lines)
            # 识别 hub nodes
            hub_nodes = find_hub_process(entity_list)  # 你可能需要根据实际函数参数调整
            # 收集与 hub nodes 相关的边
            hub_src, hub_dst, hub_t, hub_msg = [], [], [], []
            for ev in window_events:
                src = int(ev[1])
                dst = int(ev[4])
                if src in hub_nodes or dst in hub_nodes:
                    hub_src.append(src)
                    hub_dst.append(dst)
                    hub_t.append(int(ev[5]))
                    hub_msg.append(
                        torch.cat([
                            torch.from_numpy(node2higvec[src]),
                            rel2vec[ev[2]],
                            torch.from_numpy(node2higvec[dst])
                        ])
                    )
            if hub_src:
                dataset = TemporalData()
                dataset.src = torch.tensor(hub_src, dtype=torch.long)
                dataset.dst = torch.tensor(hub_dst, dtype=torch.long)
                dataset.t = torch.tensor(hub_t, dtype=torch.long)
                dataset.msg = torch.vstack(hub_msg)
                torch.save(dataset, f"{graphs_dir}/hub_graph_{day}_{window_idx}.TemporalData.simple")
            window_idx += 1
        # 新窗口
        current_window_start = ts
        current_window_end = current_window_start + window_size_ns
        window_events = []

# 处理最后一个窗口
if window_events:
    data_lines = []
    for ev in window_events:
        data_lines.append(f"{ev[1]} {ev[4]} {ev[2]}")
    entity_list = get_entity_list(data_lines)
    relation_list = get_relation_list(data_lines)
    hub_nodes = find_hub_process(entity_list)
    hub_src, hub_dst, hub_t, hub_msg = [], [], [], []
    for ev in window_events:
        src = int(ev[1])
        dst = int(ev[4])
        if src in hub_nodes or dst in hub_nodes:
            hub_src.append(src)
            hub_dst.append(dst)
            hub_t.append(int(ev[5]))
            hub_msg.append(
                torch.cat([
                    torch.from_numpy(node2higvec[src]),
                    rel2vec[ev[2]],
                    torch.from_numpy(node2higvec[dst])
                ])
            )
    if hub_src:
        dataset = TemporalData()
        dataset.src = torch.tensor(hub_src, dtype=torch.long)
        dataset.dst = torch.tensor(hub_dst, dtype=torch.long)
        dataset.t = torch.tensor(hub_t, dtype=torch.long)
        dataset.msg = torch.vstack(hub_msg)
        torch.save(dataset, f"{graphs_dir}/hub_graph_{day}_{window_idx}.TemporalData.simple")