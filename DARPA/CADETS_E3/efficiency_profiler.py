#!/usr/bin/env python3
"""
Measure CADETS_E3 training and detection efficiency.

Metrics include:
  - overall training duration
  - training duration per epoch
  - detection duration median across time windows
  - detection duration maximum across time windows
  - peak CPU memory usage
  - peak GPU memory usage (if CUDA is available)

This script invokes existing training/test functions from the project and
measures their runtime and memory behavior.
"""

import argparse
import importlib
import json
import os
import sys
import threading
import time
from collections import defaultdict
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

MODULE_LOAD_ORDER = [
    'config',
    'kairos_utils',
    'model',
    'train',
    'test',
]

DETECTION_GRAPH_NAMES = ['graph_4_6', 'graph_4_7', 'graph_4_12', 'graph_4_13']
DETECTION_GRAPH_INDICES = [3, 4, 8, 9]


class MemorySampler(threading.Thread):
    def __init__(self, interval: float = 0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_rss = 0
        self._running = False
        self._process = psutil.Process(os.getpid()) if psutil is not None else None

    def run(self) -> None:
        if self._process is None:
            return
        self._running = True
        while self._running:
            try:
                rss = self._process.memory_info().rss
                self.peak_rss = max(self.peak_rss, rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> None:
        self._running = False


class GPUMemoryTracker:
    def __init__(self):
        try:
            import torch
            self.torch = torch
            self.enabled = torch.cuda.is_available()
            if self.enabled:
                self.device = torch.cuda.current_device()
            else:
                self.device = None
        except ImportError:
            self.torch = None
            self.enabled = False
            self.device = None

    def reset(self) -> None:
        if self.enabled:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self) -> int:
        if self.enabled:
            return int(self.torch.cuda.max_memory_allocated(self.device))
        return 0


def reload_project_modules() -> Dict[str, Any]:
    modules: Dict[str, Any] = {}
    for name in MODULE_LOAD_ORDER:
        if name in sys.modules:
            modules[name] = importlib.reload(sys.modules[name])
        else:
            modules[name] = importlib.import_module(name)
    return modules


def _format_bytes(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f'{size:.2f}{unit}'
        size /= 1024
    return f'{size:.2f}PB'


def measure_training(train_mod: Any, config_mod: Any, save_model_path: Optional[str] = None) -> Tuple[Dict[str, Any], Tuple[Any, Any, Any, Any]]:
    sampler = MemorySampler() if psutil is not None else None
    gpu_tracker = GPUMemoryTracker()

    if sampler:
        sampler.start()
    if gpu_tracker.enabled:
        gpu_tracker.reset()

    start_time = time.perf_counter()
    train_data = train_mod.load_train_data()
    node_feat_size = int(train_data[0].msg.size(-1))
    memory, gnn, link_pred, optimizer, neighbor_loader = train_mod.init_models(node_feat_size=node_feat_size)

    epoch_durations: List[float] = []
    epoch_losses: List[float] = []
    for epoch in range(1, int(config_mod.epoch_num) + 1):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        for i, graph in enumerate(train_data):
            loss = train_mod.train(
                train_data=graph,
                idx=i,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader,
            )
            epoch_loss += loss
        epoch_end = time.perf_counter()
        epoch_durations.append(epoch_end - epoch_start)
        epoch_losses.append(epoch_loss / len(train_data) if len(train_data) > 0 else 0.0)

    total_duration = time.perf_counter() - start_time
    if sampler:
        sampler.stop()
        sampler.join(timeout=1.0)
    gpu_peak = gpu_tracker.peak() if gpu_tracker.enabled else 0

    model_tuple = (memory, gnn, link_pred, neighbor_loader)
    if save_model_path is not None:
        try:
            import torch
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            torch.save(model_tuple, save_model_path)
        except Exception:
            pass

    return (
        {
            'training_duration_seconds': total_duration,
            'training_duration_per_epoch_seconds': epoch_durations,
            'training_average_epoch_loss': float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0,
            'peak_cpu_rss_bytes': sampler.peak_rss if sampler else None,
            'peak_gpu_bytes': gpu_peak if gpu_tracker.enabled else None,
            'saved_model_path': save_model_path,
        },
        model_tuple,
    )


def measure_detection(test_mod: Any, kairos_utils_mod: Any, model_tuple: Tuple[Any, Any, Any, Any], config_mod: Any, output_dir: Optional[str] = None) -> Dict[str, Any]:
    sampler = MemorySampler() if psutil is not None else None
    gpu_tracker = GPUMemoryTracker()

    if sampler:
        sampler.start()
    if gpu_tracker.enabled:
        gpu_tracker.reset()

    cur, _ = kairos_utils_mod.init_database_connection()
    nodeid2msg = kairos_utils_mod.gen_nodeid2msg(cur=cur)
    graph_list = test_mod.load_data()

    memory, gnn, link_pred, neighbor_loader = model_tuple

    all_window_durations: List[float] = []
    graph_runtime_report: Dict[str, Dict[str, Any]] = {}

    for graph_name, graph_index in zip(DETECTION_GRAPH_NAMES, DETECTION_GRAPH_INDICES):
        inference_data = graph_list[graph_index]
        graph_output_dir = os.path.join(output_dir or config_mod.artifact_dir, f'detect_{graph_name}')
        os.makedirs(graph_output_dir, exist_ok=True)

        graph_start = time.perf_counter()
        time_with_loss = test_mod.test(
            inference_data=inference_data,
            idx=graph_index,
            memory=memory,
            gnn=gnn,
            link_pred=link_pred,
            neighbor_loader=neighbor_loader,
            nodeid2msg=nodeid2msg,
            path=graph_output_dir,
        )
        graph_end = time.perf_counter()

        durations = [float(info['costed_time']) for info in time_with_loss.values() if 'costed_time' in info]
        graph_runtime_report[graph_name] = {
            'graph_duration_seconds': graph_end - graph_start,
            'window_count': len(durations),
            'window_durations_seconds': durations,
            'window_duration_median_seconds': float(median(durations)) if durations else 0.0,
            'window_duration_max_seconds': max(durations) if durations else 0.0,
        }
        all_window_durations.extend(durations)

    total_duration = sum(r['graph_duration_seconds'] for r in graph_runtime_report.values())
    if sampler:
        sampler.stop()
        sampler.join(timeout=1.0)
    gpu_peak = gpu_tracker.peak() if gpu_tracker.enabled else 0

    return {
        'detection_total_duration_seconds': total_duration,
        'detection_window_duration_median_seconds': float(median(all_window_durations)) if all_window_durations else 0.0,
        'detection_window_duration_max_seconds': max(all_window_durations) if all_window_durations else 0.0,
        'detection_graph_report': graph_runtime_report,
        'peak_cpu_rss_bytes': sampler.peak_rss if sampler else None,
        'peak_gpu_bytes': gpu_peak if gpu_tracker.enabled else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Efficiency profiler for CADETS_E3.')
    parser.add_argument('--output', type=str, default='artifact/efficiency_report.json')
    parser.add_argument('--evaluate-detection', action='store_true',
                        help='Run detection profiling after training profiling.')
    args = parser.parse_args()

    project_modules = reload_project_modules()
    config_mod = project_modules['config']
    kairos_utils_mod = project_modules['kairos_utils']
    train_mod = project_modules['train']
    test_mod = project_modules['test']

    report: Dict[str, Any] = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'system': {
            'python_executable': sys.executable,
            'psutil_available': psutil is not None,
        },
    }

    print('Starting training efficiency profiling...')
    report['training'], trained_model_tuple = measure_training(
        train_mod,
        config_mod,
        save_model_path=os.path.join(config_mod.artifact_dir, 'efficiency_trained_model.pt'),
    )

    if args.evaluate_detection:
        print('Starting detection efficiency profiling...')
        report['detection'] = measure_detection(
            test_mod,
            kairos_utils_mod,
            trained_model_tuple,
            config_mod,
        )

    if psutil is not None:
        report['training']['peak_cpu_rss_human'] = _format_bytes(report['training']['peak_cpu_rss_bytes']) if report['training']['peak_cpu_rss_bytes'] else None
        if 'detection' in report:
            report['detection']['peak_cpu_rss_human'] = _format_bytes(report['detection']['peak_cpu_rss_bytes']) if report['detection']['peak_cpu_rss_bytes'] else None

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'Efficiency report written to {args.output}')


if __name__ == '__main__':
    main()
