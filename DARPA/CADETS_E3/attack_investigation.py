import os
from collections import deque, defaultdict

from graphviz import Digraph
import networkx as nx
import datetime
import community.community_louvain as community_louvain
from tqdm import tqdm

from config import *
from kairos_utils import *
from find_hub_nodes import find_hub_nodes as detect_hub_process


# Some common path abstraction for visualization
replace_dic = {
    '/run/shm/': '/run/shm/*',
    '/home/admin/.cache/mozilla/firefox/': '/home/admin/.cache/mozilla/firefox/*',
    '/home/admin/.mozilla/firefox': '/home/admin/.mozilla/firefox*',
    '/data/replay_logdb/': '/data/replay_logdb/*',
    '/home/admin/.local/share/applications/': '/home/admin/.local/share/applications/*',
    '/usr/share/applications/': '/usr/share/applications/*',
    '/lib/x86_64-linux-gnu/': '/lib/x86_64-linux-gnu/*',
    '/proc/': '/proc/*',
    '/stat': '*/stat',
    '/etc/bash_completion.d/': '/etc/bash_completion.d/*',
    '/usr/bin/python2.7': '/usr/bin/python2.7/*',
    '/usr/lib/python2.7': '/usr/lib/python2.7/*',
}


def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    return path_name


def load_window_events(window_paths):
    anomaly_events = []
    for path in window_paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    jdata = eval(line)
                except Exception:
                    continue
                if isinstance(jdata, dict) and 'loss' in jdata:
                    anomaly_events.append(jdata)
    return anomaly_events


def extract_suspicious_nodes(window_paths, zscore_thresh=1.0, percentile_thresh=90):
    anomaly_events = load_window_events(window_paths)
    losses = [float(event['loss']) for event in anomaly_events if 'loss' in event]
    if len(losses) == 0:
        return set(), anomaly_events, None, None
    thr = max(mean(losses) + zscore_thresh * std(losses), np.percentile(losses, percentile_thresh))
    suspicious_nodes = set()
    min_ts = None
    max_ts = None
    for event in anomaly_events:
        if 'time' not in event:
            continue
        ts = int(event['time'])
        if min_ts is None or ts < min_ts:
            min_ts = ts
        if max_ts is None or ts > max_ts:
            max_ts = ts
        if float(event['loss']) >= thr:
            suspicious_nodes.add(str(event['srcmsg']))
            suspicious_nodes.add(str(event['dstmsg']))
    if min_ts is None:
        min_ts = 0
        max_ts = 0
    return suspicious_nodes, anomaly_events, min_ts, max_ts


def build_raw_event_graph(cur, include_edge_types=None):
    if include_edge_types is None:
        include_edge_types = include_edge_type
    cur.execute("SELECT * FROM event_table ORDER BY timestamp_rec;")
    rows = cur.fetchall()
    forward_adj = defaultdict(list)
    backward_adj = defaultdict(list)
    raw_events = []
    for row in rows:
        if len(row) < 6:
            continue
        src_msg = str(row[1])
        rel = row[2]
        dst_msg = str(row[4])
        try:
            ts = int(row[5])
        except Exception:
            continue
        if include_edge_types is None or rel in include_edge_types:
            forward_adj[src_msg].append((dst_msg, rel, ts))
            backward_adj[dst_msg].append((src_msg, rel, ts))
            raw_events.append((src_msg, dst_msg, rel, ts))
    return forward_adj, backward_adj, raw_events


def bidirectional_causal_expand(seeds, forward_adj, backward_adj, min_ts, max_ts, delta_ns=None, max_depth=4, max_nodes=500):
    if delta_ns is None:
        delta_ns = time_window_size
    forward_edges = set()
    backward_edges = set()
    forward_nodes = set(seeds)
    backward_nodes = set(seeds)

    def _search_forward(start_node):
        queue = deque([(start_node, min_ts - delta_ns, 0)])
        visited = set()
        while queue:
            node, last_ts, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for dst, rel, ts in forward_adj.get(node, []):
                if ts < last_ts or ts < min_ts - delta_ns or ts > max_ts + delta_ns:
                    continue
                edge = (node, dst, rel, ts)
                if edge in forward_edges:
                    continue
                forward_edges.add(edge)
                forward_nodes.add(dst)
                if len(forward_nodes) > max_nodes:
                    return
                if (dst, ts, depth + 1) not in visited:
                    visited.add((dst, ts, depth + 1))
                    queue.append((dst, ts, depth + 1))

    def _search_backward(start_node):
        queue = deque([(start_node, max_ts + delta_ns, 0)])
        visited = set()
        while queue:
            node, last_ts, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for src, rel, ts in backward_adj.get(node, []):
                if ts > last_ts or ts < min_ts - delta_ns or ts > max_ts + delta_ns:
                    continue
                edge = (src, node, rel, ts)
                if edge in backward_edges:
                    continue
                backward_edges.add(edge)
                backward_nodes.add(src)
                if len(backward_nodes) > max_nodes:
                    return
                if (src, ts, depth + 1) not in visited:
                    visited.add((src, ts, depth + 1))
                    queue.append((src, ts, depth + 1))

    for seed in seeds:
        _search_forward(seed)
        _search_backward(seed)
        if len(forward_nodes) + len(backward_nodes) > max_nodes:
            break

    return forward_edges, backward_edges


def build_candidate_subgraph(forward_edges, backward_edges, anomaly_events):
    candidate_edges = set(forward_edges) | set(backward_edges)
    for event in anomaly_events:
        if 'srcmsg' in event and 'dstmsg' in event and 'edge_type' in event and 'time' in event:
            candidate_edges.add((str(event['srcmsg']), str(event['dstmsg']), str(event['edge_type']), int(event['time'])))
    all_nodes = sorted({src for src, _, _, _ in candidate_edges} | {dst for _, dst, _, _ in candidate_edges})
    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
    entity_list = [(idx, node) for idx, node in enumerate(all_nodes)]
    relation_list = [[node_to_index[src], node_to_index[dst], rel, ts] for src, dst, rel, ts in sorted(candidate_edges, key=lambda x: x[3])]
    return entity_list, relation_list, node_to_index


def reconstruct_attack_chain(entity_list, relation_list, seed_indices=None):
    if seed_indices is None:
        seed_indices = {idx for idx, _ in entity_list}
    best_path = {idx: [idx] for idx in seed_indices}
    best_time = {idx: 0 for idx in seed_indices}
    edges = sorted(relation_list, key=lambda x: x[3])
    for src, dst, rel, ts in edges:
        if src not in best_path:
            continue
        if ts < best_time[src]:
            continue
        candidate_path = best_path[src] + [dst]
        if len(candidate_path) > len(best_path.get(dst, [])):
            best_path[dst] = candidate_path
            best_time[dst] = ts
    longest_chain = []
    for path in best_path.values():
        if len(path) > len(longest_chain):
            longest_chain = path
    return longest_chain


def partition_attack_chain(chain, scn_indices):
    stage_positions = [idx for idx, node in enumerate(chain) if node in scn_indices]
    if not stage_positions:
        return [{'stage': 1, 'nodes': chain, 'anchor': None}]
    stages = []
    start = 0
    for stage_idx, pos in enumerate(stage_positions, start=1):
        stages.append({'stage': stage_idx, 'nodes': chain[start:pos + 1], 'anchor': chain[pos]})
        start = pos + 1
    if start < len(chain):
        stages.append({'stage': len(stages) + 1, 'nodes': chain[start:], 'anchor': None})
    return stages


def visualize_attack_chain(entity_list, relation_list, chain, stages, scn_indices, output_dir):
    entity_map = {idx: msg for idx, msg in entity_list}
    stage_map = {}
    for stage in stages:
        for idx in stage['nodes']:
            stage_map[idx] = stage['stage']
    colors = ['lightblue', 'lightgreen', 'orange', 'pink', 'yellow', 'lightgrey']
    graph_path = os.path.join(output_dir, 'attack_chain')
    dot = Digraph(name='AttackChain', format='pdf')
    dot.graph_attr['rankdir'] = 'LR'
    for idx, msg in entity_list:
        label = msg if len(msg) < 60 else msg[:57] + '...'
        style = 'filled' if idx in scn_indices else 'solid'
        fillcolor = colors[stage_map.get(idx, 0) % len(colors)] if idx in stage_map else 'white'
        if idx in scn_indices:
            fillcolor = 'red'
        dot.node(str(idx), label=label, shape='box', style=style, fillcolor=fillcolor)
    for src, dst, rel, ts in relation_list:
        edge_label = f'{rel}@{ts}'
        dot.edge(str(src), str(dst), label=edge_label)
    os.makedirs(output_dir, exist_ok=True)
    dot.render(graph_path, view=False)
    return graph_path + '.pdf'


def create_community_graph(window_paths, threshold_multiplier=1.5):
    gg = nx.DiGraph()
    all_edges = []
    anomaly_events = load_window_events(window_paths)
    losses = [float(event['loss']) for event in anomaly_events if 'loss' in event]
    if len(losses) == 0:
        return gg, {}
    thr = mean(losses) + threshold_multiplier * std(losses)
    for event in sorted(anomaly_events, key=lambda x: float(x['loss']), reverse=True):
        if float(event['loss']) > thr:
            src_hash = str(hashgen(replace_path_name(event['srcmsg'])))
            dst_hash = str(hashgen(replace_path_name(event['dstmsg'])))
            gg.add_edge(src_hash, dst_hash, loss=event['loss'], srcmsg=event['srcmsg'], dstmsg=event['dstmsg'], edge_type=event['edge_type'], time=event['time'])
    if gg.number_of_nodes() == 0:
        return gg, {}
    partition = community_louvain.best_partition(gg.to_undirected())
    return gg, partition


def visualize_community_graphs(gg, partition, output_dir):
    if gg.number_of_nodes() == 0:
        return []
    communities = defaultdict(nx.DiGraph)
    for node, part in partition.items():
        communities[part] = nx.DiGraph()
    for u, v in gg.edges():
        community_id = partition.get(u)
        if community_id is not None:
            communities[community_id].add_edge(u, v)
    os.makedirs(output_dir, exist_ok=True)
    rendered = []
    for c_id, graph in communities.items():
        dot = Digraph(name=f'community_{c_id}', format='pdf')
        dot.graph_attr['rankdir'] = 'LR'
        for u, v in graph.edges():
            edge_data = gg.edges[u, v]
            srcmsg = edge_data.get('srcmsg', '')
            dstmsg = edge_data.get('dstmsg', '')
            edge_label = edge_data.get('edge_type', '')
            dot.node(u, label=str(srcmsg)[:50])
            dot.node(v, label=str(dstmsg)[:50])
            dot.edge(u, v, label=edge_label)
        graph_path = os.path.join(output_dir, f'community_{c_id}')
        dot.render(graph_path, view=False)
        rendered.append(graph_path + '.pdf')
    return rendered


def main():
    os.makedirs(os.path.join(artifact_dir, 'graph_visual'), exist_ok=True)
    suspicious_nodes, anomaly_events, min_ts, max_ts = extract_suspicious_nodes(attack_list)
    if not suspicious_nodes:
        print('No suspicious nodes found in attack windows.')
        return
    cur, _ = init_database_connection()
    forward_adj, backward_adj, raw_events = build_raw_event_graph(cur)
    forward_edges, backward_edges = bidirectional_causal_expand(suspicious_nodes, forward_adj, backward_adj, min_ts, max_ts)
    entity_list, relation_list, node_to_index = build_candidate_subgraph(forward_edges, backward_edges, anomaly_events)
    hub_indices = detect_hub_process(entity_list, relation_list)
    seed_indices = {node_to_index[node] for node in suspicious_nodes if node in node_to_index}
    if not seed_indices:
        seed_indices = None
    attack_chain = reconstruct_attack_chain(entity_list, relation_list, seed_indices=seed_indices)
    stages = partition_attack_chain(attack_chain, set(hub_indices))
    chain_path = visualize_attack_chain(entity_list, relation_list, attack_chain, stages, set(hub_indices), os.path.join(artifact_dir, 'graph_visual'))
    print(f'Attack chain reconstructed and visualized at {chain_path}')
    gg, partition = create_community_graph(attack_list)
    community_paths = visualize_community_graphs(gg, partition, os.path.join(artifact_dir, 'graph_visual'))
    print('Community subgraph visualizations:', community_paths)


# Users should manually put the detected anomalous time windows here
attack_list = [
    artifact_dir+'/graph_4_6/2018-04-06_11-18-26-126177915~2018-04-06_11-33-35-116170745.txt',
    artifact_dir+'/graph_4_6/2018-04-06_11-33-35-116170745~2018-04-06_11-48-42-606135188.txt',
    artifact_dir+'/graph_4_6/2018-04-06_11-48-42-606135188~2018-04-06_12-03-50-186115455.txt',
    artifact_dir+'/graph_4_6/2018-04-06_12-03-50-186115455~2018-04-06_14-01-32-489584227.txt',
]


if __name__ == "__main__":
    main()



