import logging

import torch

from kairos_utils import *
from config import *
import networkx as nx
from collections import defaultdict
import numpy as np
# Setting for logging
logger = logging.getLogger("anomalous_queue_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'anomalous_queue.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def cal_anomaly_loss(loss_list, edge_list):
    if len(loss_list) != len(edge_list):
        print("error!")
        return 0
    count = 0
    loss_sum = 0
    loss_std = std(loss_list)
    loss_mean = mean(loss_list)
    edge_set = set()
    node_set = set()

    thr = loss_mean + 0.6* loss_std

    logger.info(f"thr:{thr}")

    for i in range(len(loss_list)):
        if loss_list[i] > thr:
            count += 1
            src_node = edge_list[i][0]
            dst_node = edge_list[i][1]
            loss_sum += loss_list[i]

            node_set.add(src_node)
            node_set.add(dst_node)
            edge_set.add(edge_list[i][0] + edge_list[i][1])
    if count == 0:
        return 0, 0, node_set, edge_set  # 或者返回 None，看你后续怎么处理
    return count, loss_sum / count, node_set, edge_set

def compute_IDF():
    node_IDF = {}

    file_list = []
    file_path = artifact_dir + "graph_4_3/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    file_path = artifact_dir + "graph_4_4/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    file_path = artifact_dir + "graph_4_5/"
    file_l = os.listdir(file_path)
    
    file_path = artifact_dir + "graph_4_9/"
    file_l = os.listdir(file_path)
    file_path = artifact_dir + "graph_4_10/"
    file_l = os.listdir(file_path)
    file_path = artifact_dir + "graph_4_11/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    node_set = {}
    for f_path in tqdm(file_list):
        f = open(f_path)
        for line in f:
            l = line.strip()
            jdata = eval(l)
            if jdata['loss'] > 0:
                if 'netflow' not in str(jdata['srcmsg']):
                    if str(jdata['srcmsg']) not in node_set.keys():
                        node_set[str(jdata['srcmsg'])] = {f_path}
                    else:
                        node_set[str(jdata['srcmsg'])].add(f_path)
                if 'netflow' not in str(jdata['dstmsg']):
                    if str(jdata['dstmsg']) not in node_set.keys():
                        node_set[str(jdata['dstmsg'])] = {f_path}
                    else:
                        node_set[str(jdata['dstmsg'])].add(f_path)
    for n in node_set:
        include_count = len(node_set[n])
        IDF = math.log(len(file_list) / (include_count + 1))
        #print(len(file_list) ,(include_count + 1))
        node_IDF[n] = IDF

    torch.save(node_IDF, artifact_dir + "node_IDF")
    logger.info("IDF weight calculate complete!")
    return node_IDF, file_list

# Measure the relationship between two time windows, if the returned value
# is not 0, it means there are suspicious nodes in both time windows.
def cal_set_rel(s1, s2, node_IDF, tw_list):
    def is_include_key_word(s):
        # The following common nodes don't exist in the training/validation data, but
        # will have the influences to the construction of anomalous queue (i.e. noise).
        # These nodes frequently exist in the testing data but don't contribute much to
        # the detection (including temporary files or files with random name).
        # Assume the IDF can keep being updated with the new time windows, these
        # common nodes can be filtered out.
        keywords = [
            'netflow',
            '/home/george/Drafts',
            'usr',
            'proc',
            'var',
            'cadet',
            '/var/log/debug.log',
            '/var/log/cron',
            '/home/charles/Drafts',
            '/etc/ssl/cert.pem',
            '/tmp/.31.3022e',
            '/tmp/.31.29d10',
            '/tmp/.31.29b50',
            '/home/henry/Archives',
            '/home/henry'
        ]
        flag = False
        for i in keywords:
            if i in s:
                flag = True
        return flag

    new_s = s1 & s2
    count = 0
    for i in new_s:
        if is_include_key_word(i) is True:
            node_IDF[i] = math.log(len(tw_list) / (1 + len(tw_list)))

        if i in node_IDF.keys():
            IDF = node_IDF[i]
        else:
            # Assign a high IDF for those nodes which are neither in training/validation
            # sets nor excluded node list above.
            IDF = math.log(len(tw_list) / (1))
#        print("+++++++++")
#        print(IDF,math.log(len(tw_list) * 0.9))
        if i=="'subject': 'pEja72mA'":
            print({i},{IDF})
            print(math.log(len(tw_list) * 0.9))
            

        # Compare the IDF with a rareness threshold α
        if IDF > (math.log(len(tw_list) * 0.9)):
            logger.info(f"node:{i}, IDF:{IDF}")
            count += 1
    return count

def build_weighted_graph(edge_list, edge_loss_list):
    """
    构建以异常得分为权重的有向图
    """
    G = nx.DiGraph()
    for i, (src, dst) in enumerate(edge_list):
        loss = edge_loss_list[i]
        G.add_edge(src, dst, loss=loss)
    return G

def compute_weighted_degree_centrality(G):
    """
    计算加权入度、出度中心性
    """
    indegree_dict = {}
    outdegree_dict = {}
    for node in G.nodes():
        indegree = sum(data.get('loss', 0) for _, _, data in G.in_edges(node, data=True))
        outdegree = sum(data.get('loss', 0) for _, _, data in G.out_edges(node, data=True))
        indegree_dict[node] = indegree
        outdegree_dict[node] = outdegree
    return indegree_dict, outdegree_dict

def compute_weighted_betweenness_centrality(G):
    """
    计算加权介数中心性（异常得分为权重，越高越异常）
    """
    epsilon = 1e-6
    for u, v, data in G.edges(data=True):
        data['inv_loss'] = 1.0 / (data.get('loss', 0) + epsilon)
    return nx.betweenness_centrality(G, weight='inv_loss')

def compute_weighted_pagerank(G):
    """
    计算加权PageRank
    """
    return nx.pagerank(G, weight='loss')

def find_patient_zero(edge_list, edge_loss_list, time_list=None):
    """
    找到最早出现高异常边的节点
    """
    if time_list is not None:
        # 假设 time_list 已经和 edge_list 对齐
        idx = np.argsort(time_list)
        for i in idx:
            if edge_loss_list[i] > 0:
                return edge_list[i][0]
    else:
        for i, loss in enumerate(edge_loss_list):
            if loss > 0:
                return edge_list[i][0]
    return None

def find_super_spreaders(outdegree_dict, topk=3):
    """
    出边异常得分总和极高的节点
    """
    return sorted(outdegree_dict, key=outdegree_dict.get, reverse=True)[:topk]

def find_sink_targets(indegree_dict, outdegree_dict, topk=3):
    """
    入边异常得分高但出边异常得分低的节点
    """
    scores = {n: indegree_dict.get(n, 0) - outdegree_dict.get(n, 0) for n in indegree_dict}
    return sorted(scores, key=scores.get, reverse=True)[:topk]

def anomalous_queue_construction(node_IDF, tw_list, graph_dir_path):
    history_list = []
    current_tw = {}

    file_l = os.listdir(graph_dir_path)
    index_count = 0
    for f_path in sorted(file_l):
        logger.info("**************************************************")
        logger.info(f"Time window: {f_path}")

        f = open(f"{graph_dir_path}/{f_path}")
        edge_loss_list = []
        edge_list = []
        logger.info(f'Time window index: {index_count}')

        # Figure out which nodes are anomalous in this time window
        for line in f:
            l = line.strip()
            jdata = eval(l)
            edge_loss_list.append(jdata['loss'])
            edge_list.append([str(jdata['srcmsg']), str(jdata['dstmsg'])])
            
        count, loss_avg, node_set, edge_set = cal_anomaly_loss(edge_loss_list, edge_list)

        current_tw['name'] = f_path
        current_tw['loss'] = loss_avg
        current_tw['index'] = index_count
        current_tw['nodeset'] = node_set

        # Incrementally construct the queues
        added_que_flag = False
        for hq in history_list:
            for his_tw in hq:
                if cal_set_rel(current_tw['nodeset'], his_tw['nodeset'], node_IDF, tw_list) != 0 and current_tw['name'] != his_tw['name']:
                    hq.append(copy.deepcopy(current_tw))
                    added_que_flag = True
                    break
                if added_que_flag:
                    break
        if added_que_flag is False:
            temp_hq = [copy.deepcopy(current_tw)]
            history_list.append(temp_hq)

        index_count += 1
        #print(history_list)


        logger.info(f"Average loss: {loss_avg}")
        logger.info(f"Num of anomalous edges within the time window: {count}")
        logger.info(f"Percentage of anomalous edges: {count / len(edge_list)}")
        logger.info(f"Anomalous node count: {len(node_set)}")
        logger.info(f"Anomalous edge count: {len(edge_set)}")
        logger.info("**************************************************")

    return history_list


if __name__ == "__main__":
    logger.info("Start logging.")

    node_IDF, tw_list = compute_IDF()

    # Validation date
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_5/"
   )
    torch.save(history_list, f"{artifact_dir}/graph_4_5_history_list")

    # Testing date
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_6/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_6_history_list")

    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_7/"
    )
    
    torch.save(history_list, f"{artifact_dir}/graph_4_7_history_list")
    
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_11/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_11_history_list")
    

    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_12/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_12_history_list")
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_13/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_13_history_list") 