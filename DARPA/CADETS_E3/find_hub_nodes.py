# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import re
import numpy as np
from array import array
import math

def get_indegree(entity_list,relation_list):
    # 节点编号 → 索引
    id_to_index = {node_id: idx for idx, node_id in enumerate(entity_list)}

    # 初始化入度列表
    indegree_list = [{node_id: 0} for node_id in entity_list]

    for rel in relation_list:
        node_id = int(rel[1])  # 取出节点编号

        # 跳过不存在的节点
        if node_id not in id_to_index:
            print(f"Warning: node_id {node_id} not in entity_list, skipping")
            continue

        idx = id_to_index[node_id]  # 找到该节点在 indegree_list 中的索引

        if idx == 0 and rel[2] == "EVENT_RECVFROM":
            continue
        else:
            indegree_list[idx][node_id] += 1

    return indegree_list

def get_outdegree(entity_list,relation_list):
    # 建立 节点编号 → 索引 的映射
    id_to_index = {node_id: idx for idx, node_id in enumerate(entity_list)}

    # 初始化出度列表
    outdegree_list = [{node_id: 0} for node_id in entity_list]

    for rel in relation_list:
        node_id = int(rel[0])  # 源节点编号

        # 跳过不存在的节点
        if node_id not in id_to_index:
            print(f"Warning: node_id {node_id} not in entity_list, skipping")
            continue

        idx = id_to_index[node_id]  # 获取该节点在 outdegree_list 中的索引
        outdegree_list[idx][node_id] += 1

    return outdegree_list

def get_degree_lists(entity_list, edge_list):
    """
    同时计算入度和出度
    entity_list: 列表，元素为元组 (node_id, content)
    edge_list: 列表，元素为元组 (src_id, dst_id, edge_type, time)
    返回: (indegree_list, outdegree_list)，其中每个元素是节点的度数值
    """
    # 从entity_list中提取节点ID列表
    node_ids = [node_id for node_id, _ in entity_list]
    
    # 节点编号 → 索引映射
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # 初始化入度和出度列表（直接存储度数值，而不是字典）
    indegree_list = [0] * len(node_ids)
    outdegree_list = [0] * len(node_ids)
    
    for edge in edge_list:
        src_id = edge[0]  # 已经是整数，不需要转换
        dst_id = edge[1]  # 已经是整数，不需要转换
        edge_type = edge[2]
        
        # 检查 src 和 dst 是否都在节点列表中
        if src_id not in id_to_index or dst_id not in id_to_index:
            print(f"Warning: edge ({src_id} -> {dst_id}) has node not in entity_list, skipping")
            continue
        
        src_idx = id_to_index[src_id]
        dst_idx = id_to_index[dst_id]
        
        # 出度：源节点出度 +1
        outdegree_list[src_idx] += 1
        
        # 入度：目标节点入度 +1（保留原有跳过逻辑）
        if not (dst_idx == 0 and edge_type == "EVENT_RECVFROM"):
            indegree_list[dst_idx] += 1
    
    return indegree_list, outdegree_list

def get_entity_list(data_lines):
    """
    data_lines: ["src dst type", ...]
    返回所有出现过的节点编号（int）
    """
    entity_set = set()
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            entity_set.add(int(parts[0]))
            entity_set.add(int(parts[1]))
    return list(entity_set)

def get_relation_list(data_lines):
    """
    data_lines: ["src dst type", ...]
    返回: [[src(int), dst(int), type(str)], ...]
    """
    relation_list = []
    for line in data_lines:
        parts = line.strip().split()
        #print(parts)
        if parts[2] not in  ["EVENT_CLOSE","EVENT_OPEN"]:
            src = int(parts[0])
            dst = int(parts[1])
            typ = parts[2]
            if parts[3]:
                time=int(parts[3])
            else:
                continue
            relation_list.append([src, dst, typ,time])
    return relation_list

def all_out_edges_smaller(out_edge, in_edge):
    for out_value in out_edge:
        for in_value in in_edge:
            if out_value >= in_value:
                return False
    return True

def has_smaller_element(out_edge, in_edge):
    for out_value in out_edge:
        for in_value in in_edge:
            if out_value < in_value:
                return True
    return False

def one_out_edges_smaller_than_all_in_edges(out_edge,in_edge):
    if len(out_edge)*len(in_edge) != 0:
        result = any(x < min(in_edge) for x in out_edge)
        return result
    else:
        return False

def one_in_edges_bigger_than_all_out_edges(out_edge,in_edge):
    if len(out_edge) * len(in_edge) != 0:
        result = any(x > max(out_edge) for x in in_edge)
        return result
    else:
        return False

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

def get_indegree_zero(indegree_list):
    """
    找出入度为0的节点索引
    indegree_list: 整数列表，每个元素表示对应节点的入度值
    返回: 入度为0的节点索引列表
    """
    indegree_zero = []
    for idx, indegree in enumerate(indegree_list):
        if indegree == 0:
            indegree_zero.append(idx)
    return indegree_zero

def get_outdegree_zero(outdegree_list):
    """
    找出出度为0的节点索引
    outdegree_list: 整数列表，每个元素表示对应节点的出度值
    返回: 出度为0的节点索引列表
    """
    outdegree_zero = []
    for idx, outdegree in enumerate(outdegree_list):
        if outdegree == 0:
            outdegree_zero.append(idx)
    return outdegree_zero

def construct_Adjacency_Matrix_Connected(entity_list, edge_list):
    """
    构建连接性邻接矩阵（0/1表示是否有连接）
    entity_list: 列表，元素为元组 (node_id, content)
    edge_list: 列表，元素为元组 (src_id, dst_id, edge_type, time)
    返回: 邻接矩阵
    """
    # 从entity_list中提取节点ID列表
    node_ids = [node_id for node_id, _ in entity_list]
    
    # 节点编号 → 索引映射
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    num_nodes = len(node_ids)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for edge in edge_list:
        source_id = edge[0]  # 已经是整数，不需要转换
        target_id = edge[1]  # 已经是整数，不需要转换
        
        # 跳过不存在的节点
        if source_id not in id_to_index or target_id not in id_to_index:
            print(f"Warning: edge ({source_id} -> {target_id}) has node not in entity_list, skipping")
            continue
        
        # 映射为矩阵索引
        source_idx = id_to_index[source_id]
        target_idx = id_to_index[target_id]
        
        adj_matrix[source_idx, target_idx] = 1
    
    return adj_matrix

def construct_Adjacency_Matrix_Value(entity_list, edge_list):
    """
    构建基于边类型的邻接矩阵（使用数值表示边类型）
    entity_list: 列表，元素为元组 (node_id, content)
    edge_list: 列表，元素为元组 (src_id, dst_id, edge_type, time)
    返回: 邻接矩阵
    """
    # 边类型到数值的映射
    bond2num = {
        'EVENT_WRITE': 0,
        'EVENT_READ': 1,
        'EVENT_CLOSE': 2,
        'EVENT_OPEN': 3,
        'EVENT_EXECUTE': 4,
        'EVENT_SENDTO': 5,
        'EVENT_RECVFROM': 6
    }
    
    # 从entity_list中提取节点ID列表
    node_ids = [node_id for node_id, _ in entity_list]
    
    # 节点编号 → 索引映射
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    num_nodes = len(node_ids)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for edge in edge_list:
        source_id = edge[0]  # 已经是整数，不需要转换
        target_id = edge[1]  # 已经是整数，不需要转换
        edge_type = edge[2]  # 边类型
        
        # 跳过不存在的节点
        if source_id not in id_to_index or target_id not in id_to_index:
            print(f"Warning: edge ({source_id} -> {target_id}) not in entity_list, skipping")
            continue
        
        # 转换为矩阵索引
        source_idx = id_to_index[source_id]
        target_idx = id_to_index[target_id]
        
        # 事件类型映射
        if edge_type not in bond2num:
            print(f"Warning: Unknown event type {edge_type}, skipping")
            continue
        
        edge_idx = bond2num[edge_type]
        adj_matrix[source_idx, target_idx] = edge_idx
    
    return adj_matrix

def construct_Adjacency_Matrix_Timestamp(entity_list, edge_list):
    """
    构建基于时间戳的邻接矩阵（使用时间戳值）
    entity_list: 列表，元素为元组 (node_id, content)
    edge_list: 列表，元素为元组 (src_id, dst_id, edge_type, time)
    返回: 邻接矩阵
    """
    # 从entity_list中提取节点ID列表
    node_ids = [node_id for node_id, _ in entity_list]
    
    # 节点编号 → 索引映射
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    num_nodes = len(node_ids)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    
    for edge in edge_list:
        source_id = edge[0]  # 已经是整数，不需要转换
        target_id = edge[1]  # 已经是整数，不需要转换
        timestamp = edge[3]  # 时间戳
        
        if source_id not in id_to_index or target_id not in id_to_index:
            print(f"Warning: edge ({source_id} -> {target_id}) not in entity_list, skipping")
            continue
        
        source_idx = id_to_index[source_id]
        target_idx = id_to_index[target_id]
        
        adj_matrix[source_idx, target_idx] = timestamp
    
    return adj_matrix

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

def is_increasing(lst):
    for i in range(1, len(lst)):
        if lst[i] <= lst[i - 1]:
            return False
    return True

def get_uncover_entity(entity_list,paths):
    uncover_entity = []
    for i in range(len(entity_list)):
        flag = False
        for path in paths:
            if i in path:
                flag = True
        if flag == False:
            uncover_entity.append(i)
    return uncover_entity

def check_subarray(list1,list2):
    array1 = array('i', list1)
    array2 = array('i', list2)
    is_subarray = array1.tobytes() in array2.tobytes()
    if array1 == array2:
        is_subarray = False
    return is_subarray

def check_subarray_2(list1,list2):
    array1 = array('i', list1)
    array2 = array('i', list2)
    is_subarray = array1.tobytes() in array2.tobytes()
    return is_subarray

def check_reasonable_path(reasonable_list):
    new_reasonable_list = []
    for i in range(len(reasonable_list)):
        check_flag = False
        for j in range(len(reasonable_list)):
            if check_subarray(reasonable_list[i],reasonable_list[j]):
                check_flag = True
            else:
                continue
        if not check_flag:
            new_reasonable_list.append(reasonable_list[i])
    return new_reasonable_list

def find_all_flows_reasonable(all_paths_find,adj_matrix_timestamp):
    all_paths_reasonable = []
    for small_path in all_paths_find:
        time_list = []
        for i1 in range(0, len(small_path) - 1):
            time_list.append(adj_matrix_timestamp[small_path[i1], small_path[i1 + 1]])
        check_increase = is_increasing(time_list)
        if (check_increase == True) and (len(small_path) != 1):
            all_paths_reasonable.append(small_path)
    all_paths_reasonable = check_reasonable_path(all_paths_reasonable)
    return all_paths_reasonable

def transfer_reasonable_flows_into_long(all_paths_reasonable,adj_matrix_value,adj_matrix_timestamp):
    num2bond = {0:'EVENT_WRITE',1:'EVENT_READ',2:'EVENT_CLOSE',3:'EVENT_OPEN',4:'EVENT_EXECUTE',5:'EVENT_SENDTO',6:'EVENT_RECVFROM'}
    all_paths_reasonable_long = []
    for small_path_2 in all_paths_reasonable:
        path_2 = []
        for i2 in range(0, len(small_path_2) - 1):
            edge = []
            edge.append(small_path_2[i2])
            edge.append(small_path_2[i2 + 1])
            #edge.append(num2bond[adj_matrix_value[small_path_2[i2], small_path_2[i2 + 1]] - 1])
            edge_value=adj_matrix_value[small_path_2[i2], small_path_2[i2 + 1]] - 1
            if edge_value>0:
                edge.append(num2bond[edge_value]) 
            else:
                edge.append(num2bond[6])
            edge.append(adj_matrix_timestamp[small_path_2[i2], small_path_2[i2 + 1]])
            path_2.append(edge)
        all_paths_reasonable_long.append(path_2)
    return (all_paths_reasonable_long)

def transfer_reasonable_flows_into_long_2(all_paths_reasonable,entity_list,adj_matrix_value):
    all_paths_reasonable_long_2 = []
    for small_path_2 in all_paths_reasonable:
        path_2 = []
        for i2 in range(0, len(small_path_2) - 1):
            edge = []
            edge.append(entity_list[small_path_2[i2]])
            edge.append(num2bond[adj_matrix_value[small_path_2[i2], small_path_2[i2 + 1]] - 1])
            edge.append(entity_list[small_path_2[i2 + 1]])
            path_2.append(edge)
        all_paths_reasonable_long_2.append(path_2)
    return (all_paths_reasonable_long_2)

def get_all_P(entity_list,relation_list,outdegree_zero):
    candidate_hub = {}
    all_P = {}
    for i, entity in enumerate(entity_list):
        #print(i,entity)
        entity=int(re.search(r'\((\d+),', str(entity)).group(1))
        for rel in relation_list:
            #print(entity,"+++",rel[0])
            if entity == rel[0]:
                all_P[i] = entity
                if i not in outdegree_zero:
                    candidate_hub[i] = entity
    return candidate_hub, all_P

def calcualte_candidate_hub_num(candidate_hub,all_paths_reasonable):
    candidate_hub_path_num = {}
    for index in candidate_hub.keys():
        count = 0
        for j in range(len(all_paths_reasonable)):
            if index in all_paths_reasonable[j]:
                count += 1
        candidate_hub_path_num[index] = count
    return (candidate_hub_path_num)

def check_uncover_path(path_list,hub_process_list):
    uncover_path = []
    for path1 in path_list:
        flag = False
        for x in range(len(path1)):
            if path1[x] in hub_process_list:
                flag = True
        if not flag:
            uncover_path.append(path1)
    return uncover_path

def get_top_k_items(dictionary, k):
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:k]

def get_top_k_keys(dictionary, k):
    sorted_keys = sorted(dictionary.keys(), key=lambda x: (dictionary[x], x), reverse=True)
    return sorted_keys[:k]

def get_hub_process_path(hub_process_index,path_list):
    hub_process_path_list = []
    for path_small in path_list:
        if hub_process_index in path_small:
            hub_process_path_list.append(path_small)
    return hub_process_path_list

def find_closest_number_index(number_list, target):
    min_difference = float('inf')
    closest_index = None
    for index, number in enumerate(number_list):
        difference = abs(number - target)
        if difference < min_difference or (difference == min_difference and index < closest_index):
            min_difference = difference
            closest_index = index
    return closest_index

def get_sum_degree(indegree_list,outdegree_list):
    sum_degree_list = {}
    for i in range(len(indegree_list)):
        sum_degree = list(indegree_list[i].values())[0] + list(outdegree_list[i].values())[0]
        sum_degree_list[i] = sum_degree
    return sum_degree_list

def overlap_percentage(set1, set2, overlap_percent_threshold):
    intersection = set1.intersection(set2)
    smaller_set = set1 if len(set1) <= len(set2) else set2
    bigger_set = set1 if len(set1) >= len(set2) else set2
    if len(smaller_set) == len(bigger_set) == 0:
        overlap_percent = 0
    else:
        overlap_percent = (len(intersection) / len(bigger_set)) * 100
    return overlap_percent >= overlap_percent_threshold

def get_percentage_number(whole_node_num,hub_node_num):
    data = np.arange(whole_node_num)
    if hub_node_num == 2:
        percentiles = np.percentile(data, [0, 50])
    elif hub_node_num == 3:
        percentiles = np.percentile(data, [0, 33.3, 66.7])
    elif hub_node_num == 4:
        percentiles = np.percentile(data, [0, 25, 50, 75])
    elif hub_node_num == 5:
        percentiles = np.percentile(data, [0, 20, 40, 60, 80])
    percentiles = np.round(percentiles).astype(int)
    return percentiles

def find_closest_process_index(entity_list,index):
    closest_index1 = -1
    closest_index2 = -1
    for i1 in range(index - 1, -1, -1):
        if entity_list[i1] in ["MP","TP"]:
            closest_index1 = i1
            break
    for i2 in range(index + 1, len(entity_list)):
        if entity_list[i2] in ["MP","TP"]:
            closest_index2 = i2
            break
    if closest_index1 == -1 and closest_index2 == -1:
        raise Exception("not found index")
    elif closest_index1 == -1:
        closest_index = closest_index2
    elif closest_index2 == -1:
        closest_index = closest_index1
    elif ((index-closest_index1) < (closest_index2-index)) and (closest_index1*closest_index2 >= 0):
        closest_index = closest_index1
    else:
        closest_index = closest_index2
    return closest_index

def get_closet_process(entity_list,whole_node_num,hub_node_num):
    percentilies = get_percentage_number(whole_node_num,hub_node_num)
    #print(percentilies)
    percentilies_list = percentilies.tolist()
    final_hub = []
    P_count = 0
    temp_hub = []
    for x,entity in enumerate(entity_list):
        if entity in ["MP","TP"]:
            P_count += 1
            temp_hub.append(x)
    if P_count == hub_node_num:
        return temp_hub
    else:
        for i in range(len(percentilies_list)):
            hub_index = percentilies_list[i]
            if entity_list[int(hub_index)] in ["MP","TP"]:
                correct_index = hub_index
                while correct_index in final_hub:
                    if hub_index+j1 < len(entity_list)-1:
                        correct_index = find_closest_process_index(entity_list, hub_index+j1)
                        j1 += 1
                    elif hub_index-j2 > 0:
                        correct_index = find_closest_process_index(entity_list, hub_index-j2)
                        j2 += 1
                final_hub.append(correct_index)
            else:
                correct_index = find_closest_process_index(entity_list,hub_index)
                j1 = 1
                j2 = 1
                while correct_index in final_hub:
                    if hub_index+j1 < len(entity_list)-1:
                        correct_index = find_closest_process_index(entity_list, hub_index+j1)
                        j1 += 1
                    elif hub_index-j2 > 0:
                        correct_index = find_closest_process_index(entity_list, hub_index-j2)
                        j2 += 1
                final_hub.append(correct_index)
        return sorted(final_hub)

def P_cut_4parts(my_list):
    total_elements = len(my_list)
    average_chunk_size = total_elements // 4
    chunks = []
    start = 0
    for i in range(4):
        end = start + average_chunk_size + (1 if i < total_elements % 4 else 0)
        chunks.append(list(my_list)[start:end])
        start = end
    return(chunks)

def find_hub_process(small_list):
    stream_num = 0
    target_hub = -1
    if len(small_list) > 0:
        for i in range(len(small_list)):
            if (small_list[i] in candidate_hub_path_num.keys()) and (candidate_hub_path_num[small_list[i]]!= 0):
                if candidate_hub_path_num[small_list[i]] > stream_num:
                    stream_num = candidate_hub_path_num[small_list[i]]
                    target_hub = small_list[i]
    return target_hub

def find_additional_P(pre_hub_process,candidate_hub_path_num):
    sorted_items = sorted(candidate_hub_path_num.items(), key=lambda x: x[1], reverse=False)
    key_value_list = [key for key, value in sorted_items]
    for x1 in range(len(key_value_list)):
        if key_value_list[x1] not in pre_hub_process:
            hub_index = key_value_list[x1]
    return hub_index

def fix_hub_process(pre_hub_process,candidate_hub_path_num):
    for i in range(len(pre_hub_process)):
        if pre_hub_process[i] == -1:
            new_hub = find_additional_P(pre_hub_process,candidate_hub_path_num)
            pre_hub_process[i] = new_hub
    return sorted(pre_hub_process)

def get_similar_list_old(candidate_hub_list,all_paths_reasonable,candidate_hub_path_num,x_threshold):
    for ch in candidate_hub_list:
        for ch1 in candidate_hub_list:
            if ch1 > ch:
                tuple_list1 = [tuple(sublist) for sublist in get_hub_process_path(ch, all_paths_reasonable)]
                tuple_list2 = [tuple(sublist) for sublist in
                               get_hub_process_path(ch1, all_paths_reasonable)]
                if set(tuple_list1) == set(tuple_list2):
                    if ch < ch1:
                        candidate_hub_path_num[ch1] = 0
                    elif ch > ch1:
                        candidate_hub_path_num[ch] = 0
                    else:
                        continue
                elif overlap_percentage(set(tuple_list1), set(tuple_list2), x_threshold) and ch1 != ch: 
                    if len(tuple_list1) >= len(tuple_list2):
                        candidate_hub_path_num[ch1] = 0
                    else:
                        candidate_hub_path_num[ch] = 0
                else:
                    continue
    #print(candidate_hub_path_num)

def get_similar_list(candidate_hub_list,all_paths_reasonable,x_threshold):
    similar_list = []
    for ch in candidate_hub_list:
        for ch1 in candidate_hub_list:
            if ch1 > ch:
                ch_list = set([tuple(sublist) for sublist in get_hub_process_path(ch, all_paths_reasonable)])
                ch1_list = set([tuple(sublist) for sublist in get_hub_process_path(ch1, all_paths_reasonable)])
                if overlap_percentage(ch_list, ch1_list, x_threshold):
                    if len(similar_list) == 0:
                        small_list = [ch, ch1]
                        similar_list.append(small_list)
                    else:
                        add_flag = False
                        cluster_ch_in = []
                        for i in range(len(similar_list)):
                            if ch in similar_list[i]:
                                cluster_ch_in.append(i)
                            if (ch1 in similar_list[i]) and (i not in cluster_ch_in):
                                cluster_ch_in.append(i)
                        if len(cluster_ch_in) == 2:
                            similar_list[cluster_ch_in[0]].extend(similar_list[cluster_ch_in[1]])
                            del similar_list[cluster_ch_in[1]]
                            add_flag = True
                        elif len(cluster_ch_in) == 1:
                            if ch in similar_list[cluster_ch_in[0]]:
                                similar_list[cluster_ch_in[0]].append(ch1)
                                add_flag = True
                            elif ch1 in similar_list[cluster_ch_in[0]]:
                                similar_list[cluster_ch_in[0]].append(ch)
                                add_flag = True
                        if add_flag == False:
                            small_list = [ch, ch1]
                            similar_list.append(small_list)
    similar_list_new = [list(set(small_list)) for small_list in similar_list]
    return(similar_list_new)

def choose_saved_entity_in_similar_list(similar_list,candidate_hub_path_num,all_paths_reasonable,x_threshold):
    save_list = []
    for s in similar_list:
        flow_dic = {}
        for s1 in s:
            flow_dic[s1] = candidate_hub_path_num[s1]
        sorted_items = sorted(flow_dic.items(), key=lambda x: x[1], reverse=True)
        sorted_dict = dict(sorted_items)
        sorted_list = list(sorted_dict.keys())
        unsatisfied_list = sorted_list[1:]
        for i in range(len(sorted_list)):
            has_satisfied = False
            has_unsatisfied = False
            satisfied_list = []
            for j in range(len(unsatisfied_list)):
                tuple_list1 = [tuple(sublist) for sublist in
                               get_hub_process_path(sorted_list[i], all_paths_reasonable)]
                tuple_list2 = [tuple(sublist) for sublist in
                               get_hub_process_path(unsatisfied_list[j], all_paths_reasonable)]
                if overlap_percentage(set(tuple_list1), set(tuple_list2), x_threshold):
                    has_satisfied = True
                    satisfied_list.append(unsatisfied_list[j])
                else:
                    has_unsatisfied = True
            unsatisfied_list = [item for item in unsatisfied_list if item not in satisfied_list]
            if has_satisfied:
                save_list.append(sorted_list[i])
            if not has_unsatisfied:
                break
    return (save_list)

def delete_high_similarity_node(candidate_hub_path_num,candidate_hub_list,all_paths_reasonable,x_threshold):
    
    similar_list = get_similar_list(candidate_hub_list, all_paths_reasonable, x_threshold)
    save_list = choose_saved_entity_in_similar_list(similar_list, candidate_hub_path_num, all_paths_reasonable,x_threshold)
    whole_similar_list = []
    for s in range(len(similar_list)):
        whole_similar_list = whole_similar_list + similar_list[s]
    not_save_list = [item for item in whole_similar_list if item not in save_list]
    for not_index in not_save_list:
        candidate_hub_path_num[not_index] = 0
    return candidate_hub_path_num

def get_P_list_tobe_select(unsatisfied_num,candidate_hub_path_num,all_P):
    non_zero_pairs = [(key, value) for key, value in candidate_hub_path_num.items() if value != 0]
    zero_pairs = [(key, value) for key, value in candidate_hub_path_num.items() if value == 0]
    select_p_list = []
    all_P_list = list(all_P.keys())

    for hub in candidate_hub_path_num.keys():
        if candidate_hub_path_num[hub] == 0:
            all_P_list.remove(hub)

    for hub in candidate_hub_path_num.keys():
        if candidate_hub_path_num[hub] != 0:
            select_p_list.append(hub)

    
    for hub in candidate_hub_path_num.keys():
        if hub not in select_p_list:
            select_p_list.append(hub)
            

    if len(non_zero_pairs) < 4:
        unsatisfied_num += 1
    return all_P_list,select_p_list,unsatisfied_num

def find_all_flows_with_right_timeorder(all_paths_find,adj_matrix_timestamp,adj_matrix_value):
    all_paths_reasonable = []
    for small_path in all_paths_find:
        time_list = []
        for i1 in range(0, len(small_path) - 1):
            time_list.append(adj_matrix_timestamp[small_path[i1], small_path[i1 + 1]])
        check_increase = is_increasing(time_list)
        if (check_increase == True) and (len(small_path) >= 1) and (adj_matrix_value[small_path[-2],small_path[-1]] != 4):
            all_paths_reasonable.append(small_path)
    all_paths_reasonable = check_reasonable_path(all_paths_reasonable)
    return all_paths_reasonable

def determine_graph_type(star_type_num,mix_type_num,flow_type_num):
    node_num = 0
    connect_num = 0
    for hub in hub_process:
        if hub != 0:
            node_num += 1
            all_paths_2 = []
            find_all_flows_between_two(adj_matrix_connected, 0, hub, [], all_paths_2)
            all_path_between_2 = find_all_flows_with_right_timeorder(all_paths_2,adj_matrix_timestamp,adj_matrix_value)
            if len(all_path_between_2) >= 1:
                connect_num += 1
    graph_type = 0
    if node_num == connect_num:
        graph_type = 1
        star_type_num += 1
    elif connect_num < node_num and connect_num != 0:
        graph_type = 2
        mix_type_num += 1
    elif connect_num == 0:
        graph_type = 3
        flow_type_num += 1
    return star_type_num,mix_type_num,flow_type_num

def divide_sub_graph(hub_process,all_paths_reasonable):
    sorted_hub_process = sorted(hub_process)
    sub_graph_list = [[] for _ in range(len(sorted_hub_process))]
    calculate = 0
    unarranged_path = []
    for path in all_paths_reasonable:
        add_flag = False
        for x in range(len(sorted_hub_process)):
            if sorted_hub_process[len(sorted_hub_process) - x - 1] in path and add_flag == False:
                    sub_graph_list[len(sorted_hub_process) - x - 1].append(path)
                    add_flag = True
        if add_flag == False:
            calculate += 1
            unarranged_path.append(path)

    if len(unarranged_path) > 0:
        for path1 in unarranged_path:
            index = find_closest_number_index(sorted_hub_process, path1[0])
            sub_graph_list[index].append(path1)
            calculate -= 1

    for sub in range(len(sub_graph_list)):
        sub_long = transfer_reasonable_flows_into_long(sub_graph_list[sub],adj_matrix_value,adj_matrix_timestamp)
    #print(sum(len(sub_graph) for sub_graph in sub_graph_list))
    return unarranged_path,sub_graph_list

def divide_sub_graph_2(hub_process,all_paths_reasonable,entity_list):
    sorted_hub_process = sorted(hub_process)
    sub_graph_list = [[] for _ in range(len(sorted_hub_process))]
    calculate = 0
    unarranged_path = []
    for path in all_paths_reasonable:
        add_flag = False
        for x in range(len(sorted_hub_process)):
            if sorted_hub_process[len(sorted_hub_process) - x - 1] in path and add_flag == False:
                    sub_graph_list[len(sorted_hub_process) - x - 1].append(path)
                    add_flag = True
        if add_flag == False:
            calculate += 1
            unarranged_path.append(path)
    if len(unarranged_path) > 0:
        for path1 in unarranged_path:
            index = find_closest_number_index(sorted_hub_process, path1[0])
            sub_graph_list[index].append(path1)
            calculate -= 1
    for sub in range(len(sub_graph_list)):
        sub_long = transfer_reasonable_flows_into_long_2(all_paths_reasonable,entity_list,adj_matrix_value)
    #print(sum(len(sub_graph) for sub_graph in sub_graph_list))
    return unarranged_path,sub_graph_list

def update_information(relation_list,sub_graph_list,hub_process,candidate_hub):
    relation_stage_dic = {}
    for r in range(len(relation_list)):
        relation = relation_list[r]
        edge_list = []
        edge_list.append(int(relation[0]))
        edge_list.append(int(relation[1]))
        flag = False
        for stage in range(len(sub_graph_list)):
            flag1 = False
            for path_num in range(len(sub_graph_list[stage])):
                if check_subarray_2(edge_list, sub_graph_list[stage][path_num]):
                    flag1 = True
                    flag = True
            if flag1:
                line_num = 2 + len(entity_list) + r
                relation_stage_dic[r] = stage+1
                data[line_num] = data[line_num].rstrip() + "-" + str(stage + 1) + "\n"
                with open(file_path_2, 'w', encoding='utf-8') as file:
                    file.writelines(data)
                    file.close()
        if not flag:
            P_stage_dic = arrange_stage_for_P(hub_process, candidate_hub,entity_list)
            stage = P_stage_dic[edge_list[0]]-1
            line_num = 2 + len(entity_list) + r
            relation_stage_dic[r] = stage + 1
            data[line_num] = data[line_num].rstrip() + "-" + str(stage + 1) + "\n"
            with open(file_path_2, 'w', encoding='utf-8') as file:
                file.writelines(data)
                file.close()
    for h in range(len(hub_process)):
        hub_line_num = 1 + hub_process[h]
        data[hub_line_num] = data[hub_line_num].rstrip() + "*" + "\n"
        with open(file_path_2, 'w', encoding='utf-8') as file:
            file.writelines(data)
            file.close()
    return relation_stage_dic

def calculate_standard_deviation(entity_list,hub_process,standard_difference_whole):
    percentalies = get_percentage_number(len(entity_list), len(hub_process))
    difference_list = []
    difference_list_divided = []
    difference_list_divided_square = []
    for k in range(len(hub_process)):
        difference_list.append(abs(percentalies[k] - hub_process[k]))
        difference_list_divided.append(difference_list[k] / (percentalies[1] - percentalies[0]))
        difference_list_divided_square.append(difference_list_divided[k] * difference_list_divided[k])
    square_mean = sum(difference_list_divided_square) / len(difference_list_divided_square)
    standard_difference = math.sqrt(square_mean)
    standard_difference_whole += standard_difference
    return standard_difference_whole

def calculate_degree_sum(P_list, entity_list, indegree_list, outdegree_list):
    """
    计算节点度数和
    P_list: 节点索引列表
    entity_list: 节点列表，元素为元组 (node_id, content)
    indegree_list: 整数列表，每个元素表示对应节点的入度值
    outdegree_list: 整数列表，每个元素表示对应节点的出度值
    返回: 字典，键为节点索引，值为度数和
    """
    sumdegree_list = {}
    for p in P_list:
        # 直接使用索引p获取度数值，不需要提取entity_id
        sum_degree = indegree_list[p] + outdegree_list[p]
        sumdegree_list[p] = sum_degree
    return sumdegree_list

def get_new_candidate_hub_num(candidate_hub_path_num,select_p_list):
    new_candidate_hub_path_num = {}
    for hub in candidate_hub_path_num.keys():
        if hub in select_p_list:
            new_candidate_hub_path_num[hub] = candidate_hub_path_num[hub]
    return new_candidate_hub_path_num

def give_mark(target_dic):
    trans_dic = {}
    highest_mark = len(target_dic.keys())
    now_mark = len(target_dic.keys())
    for key in target_dic.keys():
        trans_dic[key] = now_mark / highest_mark
        now_mark -= 1
    return trans_dic

def calculate_final_score(marked_sorted_sum_degree_dic,marked_sorted_new_candidate_hub_num,select_P_list):
    final_score_dic = {}
    weight = 0.43
    for p in select_P_list:
        score = weight*marked_sorted_sum_degree_dic[p] + (1-weight)*marked_sorted_new_candidate_hub_num[p]
        final_score_dic[p] = score
    return final_score_dic

def split_flow(hub_process,all_paths_reasonable):
    divided_paths_reasonable = []
    for flow in all_paths_reasonable:
        current_subflow = []
        for item in flow:
            if (item in hub_process) and (item != flow[0]) and (item != flow[-1]):
                current_subflow.append(item)
                if current_subflow not in divided_paths_reasonable:
                    divided_paths_reasonable.append(current_subflow)
                current_subflow = [item]
            else:
                current_subflow.append(item)
        if (len(current_subflow) != 0) and (current_subflow not in divided_paths_reasonable):
            divided_paths_reasonable.append(current_subflow)
    return divided_paths_reasonable

def arrange_stage_for_P(hub_process,candidate_hub,entity_list):
    P_stage_dic = {}
    stage1 = hub_process[0]
    stage2 = hub_process[1]
    stage3 = hub_process[2]
    stage4 = hub_process[3]
    entity_num = len(entity_list)
    for p in range(entity_num):
        if p < stage2:
            P_stage_dic[p] = 1
        elif stage2 <= p < stage3:
            P_stage_dic[p] = 2
        elif stage3 <= p < stage4:
            P_stage_dic[p] = 3
        elif stage4 <= p:
            P_stage_dic[p] = 4
    return P_stage_dic

def arrange_split_flow(divided_paths_reasonable,hub_process):
    sub_graph_list_2 = [[] for _ in range(len(hub_process))]
    unarranged_path_2 = []
    calculate = 0
    for divided_path in divided_paths_reasonable:
        add_flag = False
        for x in range(len(hub_process)):
            if hub_process[x] == divided_path[0] and divided_path not in sub_graph_list_2[x]:
                sub_graph_list_2[x].append(divided_path)
                add_flag = True
            elif (hub_process[x] != divided_path[0]) and (hub_process[x] == divided_path[-1] and adj_matrix_value[divided_path[-2]][divided_path[-1]] == 9):
                sub_graph_list_2[x].append(divided_path)
                add_flag = True
        if add_flag == False:
            calculate += 1
            unarranged_path_2.append(divided_path)

    if len(unarranged_path_2) > 0:
        for path1 in unarranged_path_2:
            index = find_closest_number_index(hub_process, path1[0])
            if path1 not in sub_graph_list_2[index]:
                sub_graph_list_2[index].append(path1)
                calculate -= 1
    return unarranged_path_2, sub_graph_list_2

def arrange_split_flow_2(divided_paths_reasonable,hub_process,candidate_hub,entity_list):
    P_stage_dic = arrange_stage_for_P(hub_process,candidate_hub,entity_list)
    sub_graph_list_2 = [[] for _ in range(len(hub_process))]
    unarranged_path_2 = []
    calculate = 0
    for divided_path in divided_paths_reasonable:
        add_flag = False
        for c_p in candidate_hub:
            t_stage = P_stage_dic[c_p] -1
            if divided_path[0] == c_p:
                if divided_path not in sub_graph_list_2[t_stage]:
                    sub_graph_list_2[t_stage].append(divided_path)
                    add_flag = True
            elif (c_p != divided_path[0]) and (c_p == divided_path[-1] and adj_matrix_value[divided_path[-2]][divided_path[-1]] == 9):
                if divided_path not in sub_graph_list_2[t_stage]:
                    sub_graph_list_2[t_stage].append(divided_path)
                    add_flag = True
        if add_flag == False:
            calculate += 1
            unarranged_path_2.append(divided_path)
    if len(unarranged_path_2) > 0:
        for path1 in unarranged_path_2:
            index = find_closest_number_index(hub_process, path1[0])
            if path1 not in sub_graph_list_2[index]:
                sub_graph_list_2[index].append(path1)
                calculate -= 1
    return unarranged_path_2, sub_graph_list_2

def is_sublist_continuous(sublist, mainlist):
    sub_len = len(sublist)
    main_len = len(mainlist)
    if sub_len > main_len:
        return False
    for i in range(main_len - sub_len + 1):
        if sublist == mainlist[i:i + sub_len]:
            return True
    return False

def check_entity_coverage(entity_list,sub_graph_list_2):
    uncover_list = []
    for e in range(len(entity_list)):
        cover_flag = False
        for stage in range(len(sub_graph_list_2)):
            for sub_flow in sub_graph_list_2[stage]:
                if e in sub_flow:
                    cover_flag = True
        if not cover_flag:
            uncover_list.append(e)
    return uncover_list

def settle_sub_graph_list(edge_list,sub_graph_list_2,hub_process,entity_list):
    P_stage_dic = arrange_stage_for_P(hub_process, candidate_hub,entity_list)
    sub_long = []
    for edge_info in edge_list:
        belong_stage_list = []
        belong_stage_list_hub = []
        edge = [int(edge_info[0]),int(edge_info[1])]
        for stage in range(len(sub_graph_list_2)):
            for sub_flow in sub_graph_list_2[stage]:
                if is_sublist_continuous(edge,sub_flow) and (stage not in belong_stage_list):
                    belong_stage_list.append(stage)
        if len(belong_stage_list) == 1:
            continue
        elif len(belong_stage_list) > 1:
            for i1 in belong_stage_list:
                belong_stage_list_hub.append(hub_process[i1])
            index = find_closest_number_index(belong_stage_list_hub, edge[0])
            target_hub = belong_stage_list_hub[index]
            target_stage = hub_process.index(target_hub)
            belong_stage_list.remove(target_stage)

            for delete_s in belong_stage_list:
                delete_list = []
                for sub_flow in sub_graph_list_2[delete_s]:
                    if is_sublist_continuous(edge,sub_flow):
                        delete_list.append(sub_flow)
                for x1 in delete_list:
                    sub_graph_list_2[delete_s].remove(x1)
    for i2 in range(len(sub_graph_list_2)):
        sub_long_2 = transfer_reasonable_flows_into_long_2(sub_graph_list_2[i2],entity_list,adj_matrix_value)
        sub_long.append(sub_long_2)
    return sub_graph_list_2,sub_long

def settle_sub_graph_list_2(edge_list,sub_graph_list_2,hub_process,entity_list):
    P_stage_dic = arrange_stage_for_P(hub_process, candidate_hub)
    sub_long = []
    for edge_info in edge_list:
        belong_stage_list = []
        belong_stage_list_hub = []
        edge = [int(edge_info[0]),int(edge_info[1])]

        for stage in range(len(sub_graph_list_2)):
            for sub_flow in sub_graph_list_2[stage]:
                if is_sublist_continuous(edge,sub_flow) and (stage not in belong_stage_list):
                    belong_stage_list.append(stage)
        if len(belong_stage_list) == 1:
            continue
        elif len(belong_stage_list) > 1:
            for i1 in belong_stage_list:
                belong_stage_list_hub.append(hub_process[i1])
            index = find_closest_number_index(belong_stage_list_hub, edge[0])
            target_hub = belong_stage_list_hub[index]
            target_stage = hub_process.index(target_hub)
            belong_stage_list.remove(target_stage)

            for delete_s in belong_stage_list:
                delete_list = []
                for sub_flow in sub_graph_list_2[delete_s]:
                    if is_sublist_continuous(edge,sub_flow):
                        delete_list.append(sub_flow)
                for x1 in delete_list:
                    sub_graph_list_2[delete_s].remove(x1)

    for i2 in range(len(sub_graph_list_2)):
        sub_long_2 = transfer_reasonable_flows_into_long_2(sub_graph_list_2[i2],entity_list,adj_matrix_value)
        sub_long.append(sub_long_2)
    return sub_graph_list_2,sub_long

def check_subflow_union(sub_graph_list_2,entity_list):
    union_dic = {}
    for i in range(len(sub_graph_list_2)):
        sub_flow_list1 = sub_graph_list_2[i]
        for j in range(i+1,len(sub_graph_list_2)):
            sub_flow_list2 = sub_graph_list_2[j]
            key = str(i + 1) + "+" + str(j + 1)
            for sub_flow1 in sub_flow_list1:
                for sub_flow2 in sub_flow_list2:
                    set1 = set(sub_flow1)
                    set2 = set(sub_flow2)
                    intersection = list(set1.intersection(set2))
                    if (len(intersection) > 0) and (key not in union_dic.keys()):
                        union_dic[key] = {}
                        for i1 in range(len(intersection)):
                            if intersection[i1] not in union_dic[key].keys():
                                union_dic[key][intersection[i1]] = entity_list[intersection[i1]]
                    elif (len(intersection) > 0) and (key in union_dic.keys()):
                        for i2 in range(len(intersection)):
                            if intersection[i2] not in union_dic[key].keys():
                                union_dic[key][intersection[i2]] = entity_list[intersection[i2]]
            for key in union_dic.keys():
                union_dic[key] = dict(sorted(union_dic[key].items(), key=lambda item: item[0]))
    return union_dic
def make_relation_type(relation_list):
    relation_type_dic = {}
    edge_dict = {"RD": 1, "WR": 2, "EX": 3, "UK": 4, "CD": 5, "FR": 6, "IJ": 7, "ST": 8, "RF": 9}
    for i in range(len(relation_list)):
        relation1 = relation_list[i][2]
        relation1_index = edge_dict[relation1]
        relation_type_dic[str(i)] = relation1_index
    return relation_type_dic

