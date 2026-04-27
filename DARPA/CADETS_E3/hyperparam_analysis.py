#!/usr/bin/env python3
"""
Hyperparameter impact experiment script for CADETS_E3.

This script is designed to compare the influence of the following hyperparameters:
  - node_embedding_dim
  - neighbor_size
  - edge_dim
  - time_window_size

The script dynamically reloads core modules after updating the configuration, then
runs training and optional evaluation for each hyperparameter combination.

Note: Since node_embedding_dim affects the graph feature generation, set
--regenerate-graphs when changing node_embedding_dim to rebuild the graph artifacts.
"""

import argparse
import importlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import config as cfg
import torch

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

MODULE_ORDER = [
    'config',
    'kairos_utils',
    'model',
    'train',
    'test',
    'embedding',
]

TEST_GRAPH_NAMES = ['graph_4_6', 'graph_4_7', 'graph_4_12', 'graph_4_13']
TEST_GRAPH_INDICES = [3, 4, 8, 9]  # based on test.load_data() ordering


def reload_modules():
    """Reload dependent modules after changing config parameters."""
    imported = {}
    for module_name in MODULE_ORDER:
        if module_name in sys.modules:
            imported[module_name] = importlib.reload(sys.modules[module_name])
        else:
            imported[module_name] = importlib.import_module(module_name)
    return imported


def set_hyperparams(
    node_embedding_dim: Optional[int] = None,
    neighbor_size: Optional[int] = None,
    edge_dim: Optional[int] = None,
    time_window_size: Optional[int] = None,
    epoch_num: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    if node_embedding_dim is not None:
        cfg.node_embedding_dim = node_embedding_dim
    if neighbor_size is not None:
        cfg.neighbor_size = neighbor_size
    if edge_dim is not None:
        cfg.edge_dim = edge_dim
    if time_window_size is not None:
        cfg.time_window_size = time_window_size
    if epoch_num is not None:
        cfg.epoch_num = epoch_num
    if batch_size is not None:
        cfg.BATCH = batch_size


def run_training(train_module, config_module) -> Tuple[Dict[str, float], str]:
    train_data = train_module.load_train_data()
    node_feat_size = int(train_data[0].msg.size(-1))

    memory, gnn, link_pred, optimizer, neighbor_loader = train_module.init_models(node_feat_size=node_feat_size)
    losses: List[float] = []

    for epoch in range(1, int(config_module.epoch_num) + 1):
        epoch_loss = 0.0
        for graph in train_data:
            epoch_loss += train_module.train(
                train_data=graph,
                idx=0,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader,
            )
        average_epoch_loss = epoch_loss / len(train_data)
        losses.append(average_epoch_loss)

    model_dir = os.path.join(config_module.artifact_dir, 'hyperparam_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(
        model_dir,
        f"model_ne{config_module.node_embedding_dim}_ns{config_module.neighbor_size}_ed{config_module.edge_dim}_tw{config_module.time_window_size}.pt",
    )
    torch.save([memory, gnn, link_pred, neighbor_loader], model_path)
    return {'average_training_loss': float(sum(losses) / len(losses)), 'epoch_losses': losses}, model_path


def run_evaluation(test_module, kairos_utils_module, model_tuple, config_module) -> Dict[str, float]:
    cur, _ = kairos_utils_module.init_database_connection()
    nodeid2msg = kairos_utils_module.gen_nodeid2msg(cur=cur)
    graph_list = test_module.load_data()
    metrics: Dict[str, float] = {}

    for graph_name, graph_idx in zip(TEST_GRAPH_NAMES, TEST_GRAPH_INDICES):
        inference_data = graph_list[graph_idx]
        memory, gnn, link_pred, neighbor_loader = model_tuple

        result = test_module.test(
            inference_data=inference_data,
            idx=graph_idx,
            memory=memory,
            gnn=gnn,
            link_pred=link_pred,
            neighbor_loader=neighbor_loader,
            nodeid2msg=nodeid2msg,
            path=os.path.join(config_module.artifact_dir, f"eval_{graph_name}"),
        )

        if result:
            window_losses = [float(r['loss']) for r in result.values() if 'loss' in r]
            metrics[f'{graph_name}_mean_loss'] = float(sum(window_losses) / len(window_losses)) if window_losses else 0.0
        else:
            metrics[f'{graph_name}_mean_loss'] = 0.0

    return metrics


def experiment_combinations(args):
    grid = {
        'node_embedding_dim': args.node_embedding_dim,
        'neighbor_size': args.neighbor_size,
        'edge_dim': args.edge_dim,
        'time_window_size': args.time_window_size,
    }
    keys = list(grid.keys())
    for values in __import__('itertools').product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def main():
    parser = argparse.ArgumentParser(description='CADETS_E3 hyperparameter impact experiment.')
    parser.add_argument('--node-embedding-dim', nargs='+', type=int, default=[16, 32, 64])
    parser.add_argument('--neighbor-size', nargs='+', type=int, default=[20, 40, 60])
    parser.add_argument('--edge-dim', nargs='+', type=int, default=[100, 200, 300])
    parser.add_argument('--time-window-size', nargs='+', type=int, default=[15, 30, 60],
                        help='Window length in minutes. Values are converted to nanoseconds.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--regenerate-graphs', action='store_true',
                        help='Regenerate graph artifacts when node_embedding_dim changes.')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on selected test graphs after training.')
    parser.add_argument('--output', type=str, default='artifact/hyperparam_results.json')
    args = parser.parse_args()

    results = []
    base_graph_dim = cfg.node_embedding_dim

    for combo in experiment_combinations(args):
        combo_ns = combo.copy()
        combo_ns['time_window_size'] = int(combo_ns['time_window_size']) * 60 * 1_000_000_000

        set_hyperparams(
            node_embedding_dim=combo_ns['node_embedding_dim'],
            neighbor_size=combo_ns['neighbor_size'],
            edge_dim=combo_ns['edge_dim'],
            time_window_size=combo_ns['time_window_size'],
            epoch_num=args.epochs,
            batch_size=args.batch_size,
        )

        reload_modules()

        if args.regenerate_graphs and combo_ns['node_embedding_dim'] != base_graph_dim:
            import embedding as embedding_module
            importlib.reload(embedding_module)
            print(f'Regenerating graphs for node_embedding_dim={combo_ns["node_embedding_dim"]} ...')
            embedding_module.main()
            base_graph_dim = combo_ns['node_embedding_dim']

        imported = reload_modules()
        train_module = imported['train']
        test_module = imported['test']
        kairos_utils_module = imported['kairos_utils']

        print(f"Running experiment: {combo_ns}")
        train_metrics, model_path = run_training(train_module, cfg)
        result = {
            'params': combo_ns,
            'train_metrics': train_metrics,
            'model_path': model_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        }

        if args.evaluate:
            imported = reload_modules()
            test_module = imported['test']
            kairos_utils_module = imported['kairos_utils']
            model_tuple = torch.load(model_path, map_location='cpu')
            eval_metrics = run_evaluation(test_module, kairos_utils_module, model_tuple, cfg)
            result['eval_metrics'] = eval_metrics

        results.append(result)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    print(f'Hyperparameter experiment complete. Results saved to {args.output}')


if __name__ == '__main__':
    main()
