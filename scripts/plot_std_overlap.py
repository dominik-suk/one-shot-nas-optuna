from pathlib import Path

import matplotlib.pyplot as plt
from src.utils.plot_io import safe_save

if __name__ == '__main__':
    algorithms = ['SPOS NSGA-II', 'Baseline', 'SPOS < 10M FLOPs']
    acc_mean = [86.66, 86.43, 85.07]
    acc_std = [0.82, 0.52, 0.52]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.errorbar(acc_mean, algorithms, xerr=acc_std, fmt='o', capsize=5)
    plt.xlabel('Accuracy')
    plt.grid(axis='x', linestyle='--')
    plt.show()

    safe_save(fig, Path('accuracy_variance.png'))

    algorithms = ['SPOS NSGA-II', 'Baseline', 'SPOS < 10M FLOPs']
    f1_mean = [80.13, 78.85, 77.45]
    f1_std = [1.67, 0.68, 1.09]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.errorbar(f1_mean, algorithms, xerr=f1_std, fmt='o', capsize=5)
    plt.xlabel('Macro F1')
    plt.grid(axis='x', linestyle='--')
    plt.show()

    safe_save(fig, Path('f1_variance.png'))