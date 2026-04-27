import logging
import os
import math
import copy
from tqdm import tqdm

import torch
import numpy as np

from kairos_utils import *
from config import *

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

    thr = loss_mean + 1 * loss_std

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
    return count, loss_sum / count, node_set, edge_set

def compute_IDF():
    node_IDF = {}
    node_freq = {}  # 记录每个节点出现的次数
    total_tw_count = 0  # 记录总时间窗口数

    file_list = []
    file_path = artifact_dir + "graph_4_3/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)
        total_tw_count += 1  # 计数时间窗口

    file_path = artifact_dir + "graph_4_4/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)
        total_tw_count += 1

    file_path = artifact_dir + "graph_4_5/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)
        total_tw_count += 1

    node_set = {}
    for f_path in tqdm(file_list):
        f = open(f_path)
        for line in f:
            l = line.strip()
            jdata = eval(l)
            if jdata['loss'] > 0:
                if 'netflow' not in str(jdata['srcmsg']):
                    node = str(jdata['srcmsg'])
                    if node not in node_set:
                        node_set[node] = {f_path}
                    else:
                        node_set[node].add(f_path)
                    # 更新节点频率
                    if node not in node_freq:
                        node_freq[node] = 1
                    else:
                        node_freq[node] += 1
                
                if 'netflow' not in str(jdata['dstmsg']):
                    node = str(jdata['dstmsg'])
                    if node not in node_set:
                        node_set[node] = {f_path}
                    else:
                        node_set[node].add(f_path)
                    # 更新节点频率
                    if node not in node_freq:
                        node_freq[node] = 1
                    else:
                        node_freq[node] += 1
                        
    for n in node_set:
        include_count = len(node_set[n])
        IDF = math.log(total_tw_count / (include_count + 1))
        node_IDF[n] = IDF

    torch.save(node_IDF, artifact_dir + "node_IDF")
    logger.info("IDF weight calculate complete!")
    
    # 返回新增的信息
    return node_IDF, file_list, node_freq, total_tw_count

def update_IDF(node_IDF, node_freq, total_tw_count, new_nodes):
    """
    动态更新IDF值
    :param node_IDF: 当前的IDF字典
    :param node_freq: 节点频率字典
    :param total_tw_count: 总时间窗口数
    :param new_nodes: 新时间窗口中的节点集合
    :return: 更新后的node_IDF, node_freq, total_tw_count
    """
    total_tw_count += 1  # 增加时间窗口计数
    
    # 更新节点频率
    for node in new_nodes:
        if node in node_freq:
            node_freq[node] += 1
        else:
            node_freq[node] = 1
    
    # 更新IDF值
    for node in new_nodes:
        if node in node_freq:
            include_count = node_freq[node]
            IDF = math.log(total_tw_count / (include_count + 1))
            node_IDF[node] = IDF
    
    return node_IDF, node_freq, total_tw_count

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
            '/tmp/.31.29d10'
            '/home/henry/Archives'
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
            # 如果节点在node_IDF中，则更新其IDF值
            if i in node_IDF:
                node_IDF[i] = math.log(len(tw_list) / (1 + len(tw_list)))
            continue  # 跳过常见节点

        # 处理不在node_IDF中的节点
        if i not in node_IDF:
            # 对于新出现的节点，赋予较高的IDF值
            IDF = math.log(len(tw_list) / 1)  # 假设只出现了一次
            node_IDF[i] = IDF  # 添加到字典中
        else:
            IDF = node_IDF[i]

        # Compare the IDF with a rareness threshold α
        if IDF > (math.log(len(tw_list) * 0.6)):
            logger.info(f"node:{i}, IDF:{IDF}")
            count += 1
    return count

def anomalous_queue_construction(node_IDF, tw_list, graph_dir_path, node_freq=None, total_tw_count=None):
    history_list = []
    current_tw = {}
    
    # 如果没有提供node_freq和total_tw_count，则初始化为空/0
    if node_freq is None:
        node_freq = {}
    if total_tw_count is None:
        total_tw_count = 0

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
        
        # 动态更新IDF
        node_IDF, node_freq, total_tw_count = update_IDF(
            node_IDF, node_freq, total_tw_count, node_set
        )

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

        logger.info(f"Average loss: {loss_avg}")
        logger.info(f"Num of anomalous edges within the time window: {count}")
        logger.info(f"Percentage of anomalous edges: {count / len(edge_list)}")
        logger.info(f"Anomalous node count: {len(node_set)}")
        logger.info(f"Anomalous edge count: {len(edge_set)}")
        logger.info("**************************************************")

    return history_list, node_IDF, node_freq, total_tw_count  # 返回更新后的值


if __name__ == "__main__":
    logger.info("Start logging.")

    # 获取初始IDF和频率信息
    node_IDF, tw_list, node_freq, total_tw_count = compute_IDF()

    # Validation date
    history_list, node_IDF, node_freq, total_tw_count = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_5/",
        node_freq=node_freq,
        total_tw_count=total_tw_count
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_5_history_list")

    # Testing date - 使用更新后的IDF和频率信息
    history_list, node_IDF, node_freq, total_tw_count = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_6/",
        node_freq=node_freq,
        total_tw_count=total_tw_count
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_6_history_list")

    # 继续使用更新后的IDF和频率信息
    history_list, node_IDF, node_freq, total_tw_count = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_7/",
        node_freq=node_freq,
        total_tw_count=total_tw_count
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_7_history_list")