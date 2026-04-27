from sklearn.metrics import confusion_matrix
import logging
import ast
from kairos_utils import *
from config import *
from model import *
from find_hub_nodes import *

# Setting for logging
logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'evaluation.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def classifier_evaluation(y_test, y_test_pred):
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
    logger.info(f'tn: {tn}')
    logger.info(f'fp: {fp}')
    logger.info(f'fn: {fn}')
    logger.info(f'tp: {tp}')

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)
    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fscore: {fscore}")
    logger.info(f"accuracy: {accuracy}")
    logger.info(f"auc_val: {auc_val}")
    return precision,recall,fscore,accuracy,auc_val

def ground_truth_label():
    labels = {}
    filelist = os.listdir(f"{artifact_dir}/graph_4_6")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_4_7")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_4_12")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_4_13")
    for f in filelist:
        labels[f] = 0

    attack_list = [
        '2018-04-06_11-18-26-126177915~2018-04-06_11-33-35-116170745.txt',
        '2018-04-06_11-33-35-116170745~2018-04-06_11-48-42-606135188.txt',
        '2018-04-06_11-48-42-606135188~2018-04-06_12-03-50-186115455.txt',
        '2018-04-06_12-03-50-186115455~2018-04-06_14-01-32-489584227.txt',
        '2018-04-12_13-58-04-106197386~2018-04-12_14-13-20-086172025.txt',
        '2018-04-12_14-13-20-086172025~2018-04-12_14-28-27-896153660.txt',
        '2018-04-12_14-28-27-896153660~2018-04-12_14-44-33-586131454.txt',
        '2018-04-13_09-03-46-284655916~2018-04-13_09-19-00-244634810.txt', 
        '2018-04-13_09-19-00-244634810~2018-04-13_09-34-55-274614479.txt',
        '2018-04-13_12-39-27-404372383~2018-04-13_12-55-00-074353420.txt',
        '2018-04-13_14-11-10-304245899~2018-04-13_14-27-05-734225138.txt'
    ]
    for i in attack_list:
        labels[i] = 1

    return labels

def calc_attack_edges():
    def keyword_hit(line):
        attack_nodes = [
            'vUgefal',
            '/var/log/devc',
            'nginx',
            '81.49.200.166',
            '78.205.235.65',
            '200.36.109.214',
            '139.123.0.113',
            '152.111.159.139',
            '61.167.39.128',

        ]
        flag = False
        for i in attack_nodes:
            if i in line:
                flag = True
                break
        return flag

    files = []
    attack_list = [
        '2018-04-06_11-18-26-126177915~2018-04-06_11-33-35-116170745.txt',
        '2018-04-06_11-33-35-116170745~2018-04-06_11-48-42-606135188.txt',
        '2018-04-06_11-48-42-606135188~2018-04-06_12-03-50-186115455.txt',
        '2018-04-06_12-03-50-186115455~2018-04-06_14-01-32-489584227.txt',
        '2018-04-12_13-58-04-106197386~2018-04-12_14-13-20-086172025.txt',
        '2018-04-12_14-13-20-086172025~2018-04-12_14-28-27-896153660.txt',
        '2018-04-12_14-28-27-896153660~2018-04-12_14-44-33-586131454.txt',
        '2018-04-13_09-03-46-284655916~2018-04-13_09-19-00-244634810.txt', 
        '2018-04-13_09-19-00-244634810~2018-04-13_09-34-55-274614479.txt',
        '2018-04-13_12-39-27-404372383~2018-04-13_12-55-00-074353420.txt',
        '2018-04-13_14-11-10-304245899~2018-04-13_14-27-05-734225138.txt'
        
    ]
    for f in attack_list:
        files.append(f"{artifact_dir}/graph_4_6/{f}")

    attack_edge_count = 0
    for fpath in (files):
        f = open(fpath)
        for line in f:
            if keyword_hit(line):
                attack_edge_count += 1
    logger.info(f"Num of attack edges: {attack_edge_count}")
def get_or_create_node_id(content,node_content_to_id, entity_list):
    """根据节点内容获取或创建节点ID"""
    global next_node_id
    content_str = str(content)  # 将内容转换为字符串用于比较
    
    if content_str in node_content_to_id:
        return node_content_to_id[content_str]
    else:
        node_id = next_node_id
        next_node_id += 1
        node_content_to_id[content_str] = node_id
        entity_list.append((node_id, content))
        return node_id

def find_hub_nodes(entity_list,relation_list):
    usable_graph_num=0
    x_threshold = 54
    unsatisfied_num=0
    cover_rate_whole=0
    indegree_list,outdegree_list=get_degree_lists(entity_list, relation_list)
    #indegree_list = get_indegree(entity_list, relation_list)
    #print(indegree_list)
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
    #print("all_paths_reasonable"+str(all_paths_reasonable))

    all_paths_reasonable_long = transfer_reasonable_flows_into_long(all_paths_reasonable,adj_matrix_value,adj_matrix_timestamp)
    #print("all_paths_reasonable_long" + str(all_paths_reasonable_long))
#    paths_whole.append(all_paths_reasonable_long)

    uncover_entity = get_uncover_entity(entity_list, all_paths_reasonable)

    candidate_hub, all_P = get_all_P(entity_list,relation_list,outdegree_zero)
    #print("========",candidate_hub)

    candidate_hub_path_num = calcualte_candidate_hub_num(candidate_hub,all_paths_reasonable)
    #print("candidate_hub_path_num"+str(candidate_hub_path_num))
    candidate_hub_list = candidate_hub_path_num.keys()
    uncover_path = check_uncover_path(all_paths_reasonable, candidate_hub_list)
                    
    usable_graph_num += 1
    candidate_hub_path_num = delete_high_similarity_node(candidate_hub_path_num,candidate_hub_list,all_paths_reasonable,x_threshold)
    all_P_list,select_P_list,unsatisfied_num = get_P_list_tobe_select(unsatisfied_num,candidate_hub_path_num,all_P)

    sum_degree_dic = calculate_degree_sum(select_P_list, entity_list, indegree_list, outdegree_list)
    #print("sum_degree_dic"+str(sum_degree_dic))
    new_candidate_hub_num = get_new_candidate_hub_num(candidate_hub_path_num,select_P_list)
    #print("new_candidate_hub_num"+str(new_candidate_hub_num))
    sorted_sum_degree_dic = {k: v for k, v in sorted(sum_degree_dic.items(), key=lambda item: item[1], reverse=True)}
    sorted_new_candidate_hub_num = {k: v for k, v in sorted(new_candidate_hub_num.items(), key=lambda item: item[1], reverse=True)}
    marked_sorted_sum_degree_dic = give_mark(sorted_sum_degree_dic)
    marked_sorted_new_candidate_hub_num = give_mark(sorted_new_candidate_hub_num)
    #print("marked_sorted_sum_degree_dic"+str(marked_sorted_sum_degree_dic))
    #print("marked_sorted_new_candidate_hub_num" + str(marked_sorted_new_candidate_hub_num))
    final_score = calculate_final_score(marked_sorted_sum_degree_dic,marked_sorted_new_candidate_hub_num,select_P_list)
    sorted_final_score = {k: v for k, v in sorted(final_score.items(), key=lambda item: item[1],reverse=True)}
    #print("final_score"+str(sorted_final_score))
    hub_process = sorted(get_top_k_keys(sorted_final_score, 10))
    #print(hub_process)
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

if __name__ == "__main__":
    logger.info("Start logging.")

    # Validation date
    anomalous_queue_scores = []
    history_list = torch.load(f"{artifact_dir}/graph_4_5_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                # Plus 1 to ensure anomaly score is monotonically increasing
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []

        for i in hl:
            name_list.append(i['name'])
        # logger.info(f"Constructed queue: {name_list}")
        # logger.info(f"Anomaly score: {anomaly_score}")

        anomalous_queue_scores.append(anomaly_score)
    history_list = torch.load(f"{artifact_dir}/graph_4_11_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                # Plus 1 to ensure anomaly score is monotonically increasing
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []

        for i in hl:
            name_list.append(i['name'])
        # logger.info(f"Constructed queue: {name_list}")
        # logger.info(f"Anomaly score: {anomaly_score}")

        anomalous_queue_scores.append(anomaly_score)
    logger.info(f"The largest anomaly score in validation set is: {max(anomalous_queue_scores)}\n")


    # Evaluating the testing set
    pred_label = {}

    filelist = os.listdir(f"{artifact_dir}/graph_4_6/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{artifact_dir}/graph_4_7/")
    for f in filelist:
        pred_label[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_4_12/")
    for f in filelist:
        pred_label[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_4_13/")
    for f in filelist:
        pred_label[f] = 0

    history_list = torch.load(f"{artifact_dir}/graph_4_6_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day6:
            name_list = []
            for i in hl:
                name_list.append(i['name'])

            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            
            logger.info(f"Anomaly score: {anomaly_score}")

    history_list = torch.load(f"{artifact_dir}/graph_4_7_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day7:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i]=1
            logger.info(f"Anomaly score: {anomaly_score}")
    history_list = torch.load(f"{artifact_dir}/graph_4_12_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day7:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i]=1
            logger.info(f"Anomaly score: {anomaly_score}")
    history_list = torch.load(f"{artifact_dir}/graph_4_13_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day13:
            name_list = []
            entity_list = []  # 存储节点信息 (node_id, content)
            edge_list = []    # 存储边信息 (src_id, dst_id, edge_type, time)
            node_content_to_id = {}  # 映射节点内容到新ID
            next_node_id = 0  # 下一个可用的节点ID
            for i in hl:
                name_list.append(i['name'])
                 #处理每个恶意窗口
                with open(artifact_dir+'graph_4_13/'+i['name']) as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        data = ast.literal_eval(line)
            
                        loss = data['loss']
                        if loss > i['loss']:
                            srcnode = data['srcnode']
                            dstnode = data['dstnode']
                            src_content = ast.literal_eval(data['srcmsg'])  # 解析嵌套字典
                            dst_content = ast.literal_eval(data['dstmsg'])  # 解析嵌套字典
                            
                            # 获取或创建节点ID
                            src_id = get_or_create_node_id(src_content,node_content_to_id,entity_list)
                            dst_id = get_or_create_node_id(dst_content,node_content_to_id,entity_list)
                            
                            edge_type = data['edge_type']
                            time = data['time']
                            
                            edge_list.append((src_id, dst_id, edge_type, time))
            
 
                            #print(f"Loss: {loss}")
                            #print(f"Src Node: {srcnode}")
                            #print(f"Dst Node: {dstnode}")
                            #print(f"Src Message: {src_content}")
                            #print(f"Dst Message: {dst_content}")
                            #print(f"Edge Type: {edge_type}")
                            #print(f"Time: {time}")
                            #print("-" * 50)
            print(f"实体数量: {len(entity_list)}")
            print(entity_list)
            print(f"边数量: {len(edge_list)}")
            #print(edge_list)
            
            hub_nodes = find_hub_nodes(entity_list,edge_list)
                    
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i]=1
            logger.info(f"Anomaly score: {anomaly_score}")

    # Calculate the metrics
    labels = ground_truth_label()
    y = []
    y_pred = []
    for i in labels:
        y.append(labels[i])
        y_pred.append(pred_label[i])
    classifier_evaluation(y, y_pred)
    for i in labels:
        if labels[i] == 1 and pred_label[i] == 1:
            logger.info(f"TP: {i}")  # 真阳性
        elif labels[i] == 1 and pred_label[i] == 0:
            logger.info(f"FN: {i}")  # 漏报
        elif labels[i] == 0 and pred_label[i] == 1:
            logger.info(f"FP: {i}")  # 误报