"""
Plot Results for DriftFL Paper.

Generates all figures needed for the IEEE paper:
- Fig. 5: Accuracy vs. communication rounds
- Fig. 6: Severity distribution (stacked bar chart)

Usage:
  python plot_results.py --results-dir results --output-dir figures
"""

import os
import csv
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for servers
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Install with: pip install matplotlib")


def load_csv(filepath):
    """Load a CSV file into a list of dicts."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_accuracy_over_rounds(results_dir, output_dir, task='cifar100'):
    """
    Generate Fig. 5: Test accuracy vs. communication round.

    Shows how each method's accuracy drops after drift onset (round 50)
    and how quickly it recovers. DriftFL should recover fastest.
    """
    if not HAS_MPL:
        return

    methods = ['fedavg', 'fedprox', 'driftfl', 'adwin', 'oracle']
    colors = {'fedavg': '#e74c3c', 'fedprox': '#3498db',
              'driftfl': '#2ecc71', 'adwin': '#f39c12', 'oracle': '#9b59b6'}
    labels = {'fedavg': 'FedAvg', 'fedprox': 'FedProx',
              'driftfl': 'DriftFL (Ours)', 'adwin': 'FedAvg+ADWIN',
              'oracle': 'Oracle'}

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        # Average across seeds
        all_accs = []
        for seed in range(42, 47):
            filepath = os.path.join(results_dir, f"{task}_{method}_seed{seed}.csv")
            if os.path.exists(filepath):
                data = load_csv(filepath)
                accs = [float(row['accuracy']) for row in data]
                all_accs.append(accs)

        if all_accs:
            mean_acc = np.mean(all_accs, axis=0)
            std_acc = np.std(all_accs, axis=0)
            rounds = list(range(1, len(mean_acc) + 1))

            ax.plot(rounds, mean_acc, color=colors[method],
                    label=labels[method], linewidth=2)
            ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                            color=colors[method], alpha=0.1)

    # Drift onset marker
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5,
               label='Drift onset')
    ax.annotate('Drift onset', xy=(50, ax.get_ylim()[1]),
                xytext=(55, ax.get_ylim()[1] - 2),
                fontsize=10, color='gray')

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'Post-Drift Recovery: {task.upper()}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'accuracy_over_rounds_{task}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_severity_distribution(results_dir, output_dir, task='cifar100'):
    """
    Generate Fig. 6: Client severity labels across rounds (stacked bar).

    Shows how many clients are classified as NONE/MODERATE/SEVERE at
    each round. After drift onset at round 50, you should see a jump
    in MODERATE and SEVERE counts.
    """
    if not HAS_MPL:
        return

    filepath = os.path.join(results_dir, f"{task}_driftfl_seed42.csv")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    data = load_csv(filepath)
    rounds = [int(row['round']) for row in data]
    sev_0 = [int(row['sev_0']) for row in data]
    sev_1 = [int(row['sev_1']) for row in data]
    sev_2 = [int(row['sev_2']) for row in data]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(rounds, sev_0, color='#2ecc71', label='None (s=0)', width=1)
    ax.bar(rounds, sev_1, bottom=sev_0, color='#f1c40f',
           label='Moderate (s=1)', width=1)
    bottom2 = [a + b for a, b in zip(sev_0, sev_1)]
    ax.bar(rounds, sev_2, bottom=bottom2, color='#e74c3c',
           label='Severe (s=2)', width=1)

    ax.axvline(x=50, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Number of Clients', fontsize=12)
    ax.set_title('Client Severity Distribution Over Rounds', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 101)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'severity_distribution_{task}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def compute_detection_metrics(results_dir, task='synthetic', method='driftfl'):
    """
    Compute precision, recall, F1, and latency for drift detection.

    Only meaningful on the synthetic dataset where ground truth is known.
    """
    filepath = os.path.join(results_dir, f"{task}_{method}_detection_seed42.csv")
    if not os.path.exists(filepath):
        print(f"Detection log not found: {filepath}")
        return None

    data = load_csv(filepath)

    # After drift onset
    post_drift = [r for r in data if int(r['round']) >= 50]

    true_positives = sum(1 for r in post_drift
                         if r['is_drifted'] == 'True' and int(r['severity']) > 0)
    false_positives = sum(1 for r in post_drift
                          if r['is_drifted'] == 'False' and int(r['severity']) > 0)
    false_negatives = sum(1 for r in post_drift
                          if r['is_drifted'] == 'True' and int(r['severity']) == 0)
    true_negatives = sum(1 for r in post_drift
                         if r['is_drifted'] == 'False' and int(r['severity']) == 0)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n--- Detection Quality ({method} on {task}) ---")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  F1:        {f1*100:.1f}%")
    print(f"  TP={true_positives}, FP={false_positives}, "
          f"FN={false_negatives}, TN={true_negatives}")

    return {'precision': precision, 'recall': recall, 'f1': f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()

    for task in ['cifar100', 'synthetic']:
        plot_accuracy_over_rounds(args.results_dir, args.output_dir, task)
        plot_severity_distribution(args.results_dir, args.output_dir, task)

    compute_detection_metrics(args.results_dir, 'synthetic', 'driftfl')
    compute_detection_metrics(args.results_dir, 'synthetic', 'adwin')
