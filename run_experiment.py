"""
DriftFL Main Experiment Runner.

Run this script to reproduce all results in the paper.

Usage:
  python run_experiment.py --task cifar100 --method driftfl --seed 42
  python run_experiment.py --task synthetic --method fedavg --seed 42
  python run_experiment.py --run-all  # Run everything

This script:
1. Sets up the federated environment (clients, data splits, drift)
2. Runs T=100 communication rounds
3. Logs accuracy, drift detections, and timing per round
4. Saves results to CSV for plotting
"""

import argparse
import os
import csv
import time
import copy
import torch
import numpy as np

import config
from models import get_model, count_parameters
from data_utils import ClientDataManager
from client import DriftFLClient, evaluate_model
from server import FederatedServer


def run_experiment(task, method, seed, device, results_dir='results'):
    """
    Run a single experiment configuration.

    Parameters
    ----------
    task : str
        'cifar100' or 'synthetic'.
    method : str
        'fedavg', 'fedprox', 'driftfl', 'adwin', or 'oracle'.
    seed : int
        Random seed.
    device : str
        'cuda' or 'cpu'.
    results_dir : str
        Directory to save CSV logs.

    Returns
    -------
    dict with final metrics.
    """
    os.makedirs(results_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Task: {task} | Method: {method} | Seed: {seed}")
    print(f"{'='*60}")

    # ---- Setup ----
    num_clients = config.NUM_CLIENTS if task == 'cifar100' else config.NUM_CLIENTS_SYNTHETIC
    model = get_model(task)
    params, model_bytes = count_parameters(model)
    print(f"Model: {params:,} params, {model_bytes/1e6:.2f} MB")

    # Data manager handles splitting and drift injection
    data_mgr = ClientDataManager(
        task=task,
        num_clients=num_clients,
        drift_start_round=config.DRIFT_START_ROUND,
        drift_fraction=config.DRIFT_FRACTION,
        seed=seed
    )

    # Server
    server = FederatedServer(
        model=model,
        num_clients=num_clients,
        client_fraction=config.CLIENT_FRACTION,
        method=method,
        seed=seed
    )

    # Create all clients
    clients = {}
    for k in range(num_clients):
        client_model = copy.deepcopy(model)
        loader = data_mgr.get_client_loader(k, batch_size=config.BATCH_SIZE)
        clients[k] = DriftFLClient(
            client_id=k,
            model=client_model,
            dataloader=loader,
            device=device,
            method=method
        )

    test_loader = data_mgr.get_test_loader()

    # ---- Training Loop ----
    round_results = []
    detection_log = []  # For drift detection quality analysis

    for rnd in range(1, config.NUM_ROUNDS + 1):
        t_start = time.time()

        # Inject drift if applicable
        data_mgr.maybe_inject_drift(rnd)

        # Update client dataloaders if data changed
        if data_mgr.drift_active:
            for k in range(num_clients):
                loader = data_mgr.get_client_loader(k, batch_size=config.BATCH_SIZE)
                clients[k].update_dataloader(loader)

        # Select clients for this round
        selected = server.select_clients(rnd)
        global_state = server.get_global_state_dict()

        # Client training and drift detection
        client_updates = []
        for k in selected:
            result = clients[k].fit(global_state, rnd)
            result['client_id'] = k
            client_updates.append(result)

            # Log drift detection for quality analysis
            if rnd >= config.DRIFT_START_ROUND:
                is_actually_drifted = data_mgr.is_client_drifted(k)
                detection_log.append({
                    'round': rnd,
                    'client_id': k,
                    'severity': result['severity'],
                    'is_drifted': is_actually_drifted,
                    'phi': result['phi']
                })

        # Server aggregation
        drifted_set = set(data_mgr.drifted_clients) if method == 'oracle' else None
        agg_stats = server.aggregate(
            client_updates,
            drifted_clients_ground_truth=drifted_set,
            current_round=rnd
        )

        # Evaluate global model
        accuracy = evaluate_model(server.global_model, test_loader, device)
        t_elapsed = time.time() - t_start

        round_entry = {
            'round': rnd,
            'accuracy': accuracy,
            'time_sec': t_elapsed,
            'effective_clients': agg_stats.get('effective_clients', 0),
            'sev_0': agg_stats.get('severity_counts', {}).get(0, 0),
            'sev_1': agg_stats.get('severity_counts', {}).get(1, 0),
            'sev_2': agg_stats.get('severity_counts', {}).get(2, 0),
        }
        round_results.append(round_entry)

        if rnd % 10 == 0 or rnd == 1:
            print(f"  Round {rnd:3d} | Acc: {accuracy:.2f}% | "
                  f"Time: {t_elapsed:.1f}s | "
                  f"Sev: [{round_entry['sev_0']}/{round_entry['sev_1']}/{round_entry['sev_2']}]")

    # ---- Save Results ----
    csv_path = os.path.join(results_dir, f"{task}_{method}_seed{seed}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=round_results[0].keys())
        writer.writeheader()
        writer.writerows(round_results)
    print(f"Saved round results to {csv_path}")

    # Save detection log
    if detection_log:
        det_path = os.path.join(results_dir, f"{task}_{method}_detection_seed{seed}.csv")
        with open(det_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=detection_log[0].keys())
            writer.writeheader()
            writer.writerows(detection_log)

    # Final metrics
    final_accuracy = round_results[-1]['accuracy']
    post_drift_accs = [r['accuracy'] for r in round_results
                       if r['round'] >= config.DRIFT_START_ROUND]
    avg_post_drift = np.mean(post_drift_accs) if post_drift_accs else final_accuracy

    # Detection latency
    if detection_log and method in ('driftfl', 'adwin'):
        first_detections = {}
        for entry in detection_log:
            if entry['is_drifted'] and entry['severity'] > 0:
                k = entry['client_id']
                if k not in first_detections:
                    first_detections[k] = entry['round']
        latencies = [r - config.DRIFT_START_ROUND for r in first_detections.values()
                     if r >= config.DRIFT_START_ROUND]
        avg_latency = np.mean(latencies) if latencies else float('inf')
    else:
        avg_latency = float('inf')

    summary = {
        'task': task,
        'method': method,
        'seed': seed,
        'final_accuracy': final_accuracy,
        'avg_post_drift_accuracy': avg_post_drift,
        'detection_latency': avg_latency,
    }
    print(f"\n  Final Accuracy: {final_accuracy:.2f}%")
    print(f"  Avg Post-Drift Accuracy: {avg_post_drift:.2f}%")
    if avg_latency < float('inf'):
        print(f"  Avg Detection Latency: {avg_latency:.1f} rounds")

    return summary


def run_all_experiments(device, results_dir='results'):
    """Run all combinations of tasks and methods across multiple seeds."""
    tasks = ['cifar100', 'synthetic']
    methods = ['fedavg', 'fedprox', 'driftfl', 'adwin', 'oracle']
    seeds = list(range(config.SEED, config.SEED + config.NUM_RUNS))

    all_summaries = []
    for task in tasks:
        for method in methods:
            for seed in seeds:
                summary = run_experiment(task, method, seed, device, results_dir)
                all_summaries.append(summary)

    # Save combined summary
    summary_path = os.path.join(results_dir, 'all_results_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\nSaved all summaries to {summary_path}")

    # Print aggregated results table
    print("\n" + "="*70)
    print("AGGREGATED RESULTS (mean +/- std across seeds)")
    print("="*70)
    for task in tasks:
        print(f"\n--- {task.upper()} ---")
        print(f"{'Method':<15} {'Accuracy':>15} {'Latency':>15}")
        for method in methods:
            accs = [s['avg_post_drift_accuracy'] for s in all_summaries
                    if s['task'] == task and s['method'] == method]
            if accs:
                print(f"{method:<15} {np.mean(accs):>7.2f} +/- {np.std(accs):<5.2f} "
                      f"{'N/A':>10}")


def main():
    parser = argparse.ArgumentParser(description="DriftFL Experiment Runner")
    parser.add_argument('--task', type=str, default='cifar100',
                        choices=['cifar100', 'synthetic'])
    parser.add_argument('--method', type=str, default='driftfl',
                        choices=['fedavg', 'fedprox', 'driftfl', 'adwin', 'oracle'])
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--device', type=str, default=config.DEVICE)
    parser.add_argument('--run-all', action='store_true',
                        help='Run all task/method/seed combinations')
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    if args.run_all:
        run_all_experiments(device, args.results_dir)
    else:
        run_experiment(args.task, args.method, args.seed, device, args.results_dir)


if __name__ == "__main__":
    main()
