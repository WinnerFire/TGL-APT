from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm

import numpy as np
import logging
import torch
import os
from collections import defaultdict
from config import *
from kairos_utils import *
from find_hub_nodes import *
import networkx as nx
import community as community_louvain
# Setting for logging
logger = logging.getLogger("embedding_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'embedding.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def path2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
    return l

def ip2higlist(p):
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
    return l

def list2str(l):
    s=''
    for i in l:
        s+=i
    return s

def gen_feature(cur):
    # Firstly obtain all node labels
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Construct the hierarchical representation for each node label
    node_msg_dic_list = []
    for i in tqdm(nodeid2msg.keys()):
        if type(i) == int:
            if 'netflow' in nodeid2msg[i].keys():
                higlist = ['netflow']
                higlist += ip2higlist(nodeid2msg[i]['netflow'])

            if 'file' in nodeid2msg[i].keys():
                higlist = ['file']
                higlist += path2higlist(nodeid2msg[i]['file'])

            if 'subject' in nodeid2msg[i].keys():
                higlist = ['subject']
                higlist += path2higlist(nodeid2msg[i]['subject'])
            node_msg_dic_list.append(list2str(higlist))

    # Featurize the hierarchical node labels
    FH_string = FeatureHasher(n_features=node_embedding_dim, input_type="string")
    node2higvec=[]
    for i in tqdm(node_msg_dic_list):
        #vec=FH_string.transform([i]).toarray()
        token_list = i.split()  # 将字符串拆分成字符串列表
        vec = FH_string.transform([token_list]).toarray()
        node2higvec.append(vec)
    node2higvec = np.array(node2higvec).reshape([-1, node_embedding_dim])
    torch.save(node2higvec, artifact_dir + "node2higvec")
    return node2higvec

def gen_relation_onehot():
    relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)
    rel2vec={}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i]= relvec[rel2id[i]-1]
            rel2vec[relvec[rel2id[i]-1]]=i
    torch.save(rel2vec, artifact_dir + "rel2vec")
    return rel2vec
def find_hub_nodes(entity_list,relation_list):
    usable_graph_num=0
    x_threshold = 54
    unsatisfied_num=0
    cover_rate_whole=0
    indegree_list,outdegree_list=get_degree_lists(entity_list, relation_list)
    #indegree_list = get_indegree(entity_list, relation_list)
    print(indegree_list)
    #outdegree_list = get_outdegree(entity_list, relation_list)
    #print(outdegree_list)
    indegree_zero = get_indegree_zero(indegree_list)
    outdegree_zero = get_outdegree_zero(outdegree_list)
    adj_matrix_connected = construct_Adjacency_Matrix_Connected(entity_list,relation_list)
    adj_matrix_value = construct_Adjacency_Matrix_Value(entity_list,relation_list)
    adj_matrix_timestamp = construct_Adjacency_Matrix_Timestamp(entity_list,relation_list)
    special_node = find_special_node(adj_matrix_timestamp, entity_list)
    special_start_node = find_special_start_node(adj_matrix_timestamp, entity_list)
    special_end_node = find_special_end_node(adj_matrix_timestamp, entity_list)
    indegree_zero_plus = list(set(indegree_zero + special_start_node))
    outdegree_zero_plus = list(set(outdegree_zero + special_end_node))
                    
    all_paths_find = find_all_flows(indegree_zero_plus,outdegree_zero_plus,adj_matrix_connected)

    all_paths_reasonable = find_all_flows_reasonable(all_paths_find,adj_matrix_timestamp)
    print("all_paths_reasonable"+str(all_paths_reasonable))

    all_paths_reasonable_long = transfer_reasonable_flows_into_long(all_paths_reasonable,adj_matrix_value,adj_matrix_timestamp)
    print("all_paths_reasonable_long" + str(all_paths_reasonable_long))
#    paths_whole.append(all_paths_reasonable_long)

    uncover_entity = get_uncover_entity(entity_list, all_paths_reasonable)

    candidate_hub, all_P = get_all_P(entity_list,relation_list,outdegree_zero)

    candidate_hub_path_num = calcualte_candidate_hub_num(candidate_hub,all_paths_reasonable)
    print("candidate_hub_path_num"+str(candidate_hub_path_num))
    candidate_hub_list = candidate_hub_path_num.keys()
    uncover_path = check_uncover_path(all_paths_reasonable, candidate_hub_list)
                    
    usable_graph_num += 1
    candidate_hub_path_num = delete_high_similarity_node(candidate_hub_path_num,candidate_hub_list,all_paths_reasonable,x_threshold)
    all_P_list,select_P_list,unsatisfied_num = get_P_list_tobe_select(unsatisfied_num,candidate_hub_path_num,all_P)

    sum_degree_dic = calculate_degree_sum(select_P_list, entity_list, indegree_list, outdegree_list)
    print("sum_degree_dic"+str(sum_degree_dic))
    new_candidate_hub_num = get_new_candidate_hub_num(candidate_hub_path_num,select_P_list)
    print("new_candidate_hub_num"+str(new_candidate_hub_num))
    sorted_sum_degree_dic = {k: v for k, v in sorted(sum_degree_dic.items(), key=lambda item: item[1], reverse=True)}
    sorted_new_candidate_hub_num = {k: v for k, v in sorted(new_candidate_hub_num.items(), key=lambda item: item[1], reverse=True)}
    marked_sorted_sum_degree_dic = give_mark(sorted_sum_degree_dic)
    marked_sorted_new_candidate_hub_num = give_mark(sorted_new_candidate_hub_num)
    print("marked_sorted_sum_degree_dic"+str(marked_sorted_sum_degree_dic))
    print("marked_sorted_new_candidate_hub_num" + str(marked_sorted_new_candidate_hub_num))
    final_score = calculate_final_score(marked_sorted_sum_degree_dic,marked_sorted_new_candidate_hub_num,select_P_list)
    sorted_final_score = {k: v for k, v in sorted(final_score.items(), key=lambda item: item[1],reverse=True)}
    print("final_score"+str(sorted_final_score))
    hub_process = sorted(get_top_k_keys(sorted_final_score, 10))
    print(hub_process)
    uncover_path = check_uncover_path(all_paths_reasonable, hub_process)
    if len(all_paths_reasonable) > 0:
        cover_rate = (len(all_paths_reasonable) - len(uncover_path)) / len(all_paths_reasonable)
    else:
        cover_rate = 0.0  # 或者 np.nan，看你的业务需求

    cover_rate_whole += cover_rate

    if usable_graph_num > 0:  # 同时防止后面也除以 0
        cover_rate_mean = cover_rate_whole / usable_graph_num
    else:
        cover_rate_mean = 0.0

    print(cover_rate_mean)
    
    return hub_process
                    
def gen_vectorized_graphs_old(cur, node2higvec, rel2vec, logger):
    for day in tqdm(range(2, 14)):
        start_timestamp = datetime_to_ns_time_US('2018-04-' + str(day) + ' 00:00:00')
        end_timestamp = datetime_to_ns_time_US('2018-04-' + str(day + 1) + ' 00:00:00')
        sql = """
        select * from event_table
        where
              timestamp_rec>'%s' and timestamp_rec<'%s'
               ORDER BY timestamp_rec;
        """ % (start_timestamp, end_timestamp)
        cur.execute(sql)
        events = cur.fetchall()
        logger.info(f'2018-04-{day}, events count: {len(events)}')
        edge_list = []
        for e in events:
            #print(e)
            
            edge_temp = [int(e[1]), int(e[4]), e[2], e[5]]
            if e[2] in include_edge_type:
                edge_list.append(edge_temp)
        logger.info(f'2018-04-{day}, edge list len: {len(edge_list)}')
        dataset = TemporalData()
        
        src = []
        dst = []
        msg = []
        t = []
        paths_whole = []
        windows_events = []
        window_idx = 0
        window_size_events = 3000  # 每个窗口事件数
        #---------------------------------------------------------
        # 新增两个全局变量
        event_freq = {}           # 统计事件出现的窗口次数
        whitelist_events = set()  # 存放白名单事件
        #start_time = edge_list[0][3]
        #end_time = edge_list[-1][3]
        #current_window_start = start_time
        #current_window_end = current_window_start + time_window_size
        
        
        for i in edge_list:
            src.append(int(i[0]))
            dst.append(int(i[1]))
            msg.append(
                torch.cat([torch.from_numpy(node2higvec[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec[i[1]])]))
            t.append(int(i[3]))
            ts = i[3]
            # 事件唯一标识（假设 src, dst, rel 组合能唯一确定一个事件）
            event_key = (int(i[0]), int(i[1]), i[2])
            # 跳过白名单事件
            if event_key in whitelist_events:
                continue
            windows_events.append(i)
            if len(windows_events)>= window_size_events:
                    data_lines = []
                    for ev in windows_events:
                        data_lines.append(f"{ev[0]} {ev[1]} {ev[2]} {ev[3]}")
                        
                    entity_list = get_entity_list(data_lines)
                    relation_list = get_relation_list(data_lines)
                    #print(relation_list)
                    print("gogogo")
                    hub_nodes = find_hub_nodes(entity_list,relation_list)
                    hub_flag=[]
                    src1 = int(ev[0])
                    dst1 = int(ev[1])
                    if src1 in hub_nodes or dst1 in hub_nodes:
                        hub_flag.append(1)
                    else:
                        hub_flag.append(0)
                    # ====== 新增统计事件出现频率 ======
            # 本窗口的去重事件
                    unique_events = set((int(ev[0]), int(ev[1]), ev[2]) for ev in windows_events)

                    for ekey in unique_events:
                        event_freq[ekey] = event_freq.get(ekey, 0) + 1
                # 出现满 5 次加入白名单
                        if event_freq[ekey] >= 20:
                            whitelist_events.add(ekey)
            # ====== 新增统计事件出现频率结束 ======
                    windows_events = []
                    window_idx += 1
                
                
        if windows_events:
               data_lines = []
               for ev in windows_events:
                    data_lines.append(f"{ev[0]} {ev[1]} {ev[2]}")
               entity_list = get_entity_list(data_lines)
               relation_list = get_relation_list(data_lines)
                    
               hub_nodes = find_hub_nodes(entity_list,relation_list)
               hub_flag=[]
               src1 = int(ev[0])
               dst1 = int(ev[1])
               if src1 in hub_nodes or dst1 in hub_nodes:
                    hub_flag.append(1)
               else:
                    hub_flag.append(0)
                

        dataset.src = torch.tensor(src)
        dataset.dst = torch.tensor(dst)
        dataset.t = torch.tensor(t)
        dataset.hub_flag = torch.tensor(hub_flag)
        dataset.msg = torch.vstack(msg)
        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        dataset.t = dataset.t.to(torch.long)
        dataset.hub_flag = dataset.hub_flag.to(torch.long)
        torch.save(dataset, graphs_dir + "/graph_4_" + str(day) + ".TemporalData.simple")

def gen_vectorized_graphs_2(cur, node2higvec, rel2vec, logger):
    for day in tqdm(range(2, 14)):
        start_timestamp = datetime_to_ns_time_US('2018-04-' + str(day) + ' 00:00:00')
        end_timestamp = datetime_to_ns_time_US('2018-04-' + str(day + 1) + ' 00:00:00')
        sql = """
        select * from event_table
        where
              timestamp_rec>'%s' and timestamp_rec<'%s'
               ORDER BY timestamp_rec;
        """ % (start_timestamp, end_timestamp)
        cur.execute(sql)
        events = cur.fetchall()
        logger.info(f'2018-04-{day}, events count: {len(events)}')
        edge_list = []
        for e in events:
            edge_temp = [int(e[1]), int(e[4]), e[2], e[5]]
            if e[2] in include_edge_type:
                edge_list.append(edge_temp)
        logger.info(f'2018-04-{day}, edge list len: {len(edge_list)}')
        dataset = TemporalData()
        src = []
        dst = []
        msg = []
        t = []
        
        G = nx.Graph()
        
        for i in edge_list:
            src.append(int(i[0]))
            dst.append(int(i[1]))
            msg.append(
                torch.cat([torch.from_numpy(node2higvec[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec[i[1]])]))
            t.append(int(i[3]))
            G.add_edge(i[0], i[1])
        partition = community_louvain.best_partition(G)
        communities = set(partition.values())
        key_nodes = {}
        ratio = 0.1  # 可调整比例
        max_exact_size = 3000  # 阈值，可调整
        for com in tqdm(communities):
            nodes_in_com = [node for node in partition if partition[node] == com]
            subgraph = G.subgraph(nodes_in_com)
            n = len(nodes_in_com)
            k = max(1, int(n * ratio))
            if n <= max_exact_size:
                centrality = nx.betweenness_centrality(subgraph)
            else:
        # 超大社区用度中心性或近似算法
                centrality = nx.degree_centrality(subgraph)
        # 或采样部分节点
        # sampled_nodes = random.sample(nodes_in_com, min(500, n))
        # centrality = nx.betweenness_centrality(subgraph, k=min(500, n))
            sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
            key_nodes[com] = sorted_nodes[:k]

        logger.info(f"社区划分结果: {partition}")
        logger.info(f"每个社区的关键节点: {key_nodes}")

        dataset.src = torch.tensor(src)
        dataset.dst = torch.tensor(dst)
        dataset.t = torch.tensor(t)
        dataset.msg = torch.vstack(msg)
        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        dataset.t = dataset.t.to(torch.long)
        torch.save(dataset, graphs_dir + "/graph_4_" + str(day) + ".TemporalData.simple")

def gen_vectorized_graphs(cur, node2higvec, rel2vec, logger,
                          window_size_minutes=15):
    """
    基于时间窗口的社区划分，结合中心性和IDF稀有度加权，保存：
    1. 全局稀有节点排名（idf_rank.log）
    2. 每天的关键节点（key_nodes.log）
    3. 每个窗口的图（graph_4_{day}_win_{win_id}.TemporalData.simple）
    """

    # ========= 第一次扫描：统计每个节点在多少个窗口出现 =========
    node_window_freq = defaultdict(set)
    total_windows = 0

    for day in range(2, 14):
        start_timestamp = datetime_to_ns_time_US(f'2018-04-{day} 00:00:00')
        end_timestamp = datetime_to_ns_time_US(f'2018-04-{day + 1} 00:00:00')
        sql = f"""
        select * from event_table
        where timestamp_rec>'{start_timestamp}' and timestamp_rec<'{end_timestamp}'
        ORDER BY timestamp_rec;
        """
        cur.execute(sql)
        events = cur.fetchall()

        if not events:
            continue

        # 按时间窗口分组
        window_size_ns = window_size_minutes * 60 * int(1e9)
        first_ts = events[0][5]
        window_start = first_ts - (first_ts % window_size_ns)
        edges_by_window = defaultdict(list)

        for e in events:
            ts = int(e[5])
            win_id = (ts - window_start) // window_size_ns + total_windows
            if e[2] in include_edge_type:
                edges_by_window[win_id].append([int(e[1]), int(e[4]), e[2], ts])
                node_window_freq[int(e[1])].add(win_id)
                node_window_freq[int(e[4])].add(win_id)

        total_windows += len(edges_by_window)

    # ========= 计算全局 IDF =========
    node_idf = {}
    for node, wins in node_window_freq.items():
        df = len(wins)
        if df > 0:
            node_idf[node] = math.log(total_windows / df)
        else:
            node_idf[node] = 0.0

    logger.info(f"总窗口数: {total_windows}")
    logger.info(f"样例节点IDF: {list(node_idf.items())[:10]}")

    # 保存全局稀有节点排名
    idf_rank_log = graphs_dir + "/idf_rank.log"
    with open(idf_rank_log, "w") as f:
        rare_nodes_global = sorted(node_idf.items(), key=lambda x: x[1], reverse=True)
        for node, score in rare_nodes_global:
            f.write(f"{node}\t{score:.6f}\n")
    logger.info(f"全局IDF排名已保存到: {idf_rank_log}")

    # ========= 第二次扫描：构图、社区划分、关键节点识别 =========
    key_nodes_log_path = graphs_dir + "/key_nodes.log"

    with open(key_nodes_log_path, "w") as kn_log:
        for day in tqdm(range(2, 14)):
            start_timestamp = datetime_to_ns_time_US(f'2018-04-{day} 00:00:00')
            end_timestamp = datetime_to_ns_time_US(f'2018-04-{day + 1} 00:00:00')
            sql = f"""
            select * from event_table
            where timestamp_rec>'{start_timestamp}' and timestamp_rec<'{end_timestamp}'
            ORDER BY timestamp_rec;
            """
            cur.execute(sql)
            events = cur.fetchall()
            logger.info(f'2018-04-{day}, events count: {len(events)}')

            if not events:
                kn_log.write("[]\n")
                continue

            # 按时间窗口分组
            window_size_ns = window_size_minutes * 60 * int(1e9)
            first_ts = events[0][5]
            window_start = first_ts - (first_ts % window_size_ns)
            edges_by_window = defaultdict(list)

            for e in events:
                ts = int(e[5])
                win_id = (ts - window_start) // window_size_ns
                if e[2] in include_edge_type:
                    edges_by_window[win_id].append([int(e[1]), int(e[4]), e[2], ts])

            # 保存当天关键节点集合
            day_key_nodes_set = set()
            src, dst, msg, t = [], [], [], []

            for win_id, edge_list in edges_by_window.items():
                
                G = nx.Graph()

                for i in edge_list:
                    src.append(int(i[0]))
                    dst.append(int(i[1]))
                    msg.append(
                        torch.cat([
                            torch.from_numpy(node2higvec[i[0]]),
                            rel2vec[i[2]],
                            torch.from_numpy(node2higvec[i[1]])
                        ])
                    )
                    t.append(int(i[3]))
                    G.add_edge(i[0], i[1])

                if G.number_of_nodes() == 0:
                    continue

                # Louvain社区划分
                partition = community_louvain.best_partition(G)
                communities = set(partition.values())

                ratio = 0.3
                max_exact_size = 3000

                for com in tqdm(communities):
                    nodes_in_com = [node for node in partition if partition[node] == com]
                    subgraph = G.subgraph(nodes_in_com)
                    n = len(nodes_in_com)
                    k = max(1, int(n * ratio))

                    if n <= max_exact_size:
                        centrality = nx.betweenness_centrality(subgraph)
                    else:
                        centrality = nx.degree_centrality(subgraph)

                    # 中心性 × (1 + IDF)
                    for node in centrality:
                        centrality[node] *= (1.0 + node_idf.get(node, 0.0))

                    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
                    key_nodes_window = sorted_nodes[:k]

                    day_key_nodes_set.update(key_nodes_window)

                # 保存窗口图数据
            dataset = TemporalData()
            dataset.src = torch.tensor(src, dtype=torch.long)
            dataset.dst = torch.tensor(dst, dtype=torch.long)
            dataset.t = torch.tensor(t, dtype=torch.long)
            dataset.msg = torch.vstack(msg).to(torch.float)
            torch.save(dataset, graphs_dir + f"/graph_4_{day}.TemporalData.simple")

            # 写当天关键节点到 log
            day_key_nodes_list = sorted(list(day_key_nodes_set))
            kn_log.write(str(day_key_nodes_list) + "\n")
            logger.info(f"2018-04-{day} 关键节点: {day_key_nodes_list}")


if __name__ == "__main__":
    logger.info("Start logging.")

    os.system(f"mkdir -p {graphs_dir}")

    cur, _ = init_database_connection()
    node2higvec = gen_feature(cur=cur)
    rel2vec = gen_relation_onehot()
    gen_vectorized_graphs(cur=cur, node2higvec=node2higvec, rel2vec=rel2vec, logger=logger)

