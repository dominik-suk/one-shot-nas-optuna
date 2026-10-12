# One-Shot NAS with Optuna

This software implements [Single Path One-Shot](https://arxiv.org/abs/1904.00420) weight-sharing NAS algorithm by Guo et al. with [Optuna](https://optuna.org) and using the [elasticAI.explorer](https://arxiv.org/abs/2605.30019) hardware NAS toolbox for the thesis titled 'Implementation and Evaluation of a
One-Shot Supernet Approach for Neural
Architecture Search'.

## Installation

Prerequisites:
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

To run the complete pipeline, execute the bash run script:

    $ bash scripts/.run.sh

Or execute the parts separately using the scripts inside `scripts/*.py`


## Abstract

Neural Architecture Search (NAS) aims to automate the manual, expert-driven process of designing Deep Neural Networks (DNNs). Traditional NAS methods demand immense computational power by training thousands of architectures from scratch. To overcome this cost, one-shot supernets have emerged as a paradigm where parameters are shared across architectures. 

However, despite a substantial search speed gain, sharing weights across subnets causes parameters to become entangled, possibly degrading the supernets ability to reliably rank architectures. Furthermore, research lacks wide-spread usage of suitable hyperparameter frameworks like Optuna to standardize one-shot NAS procedures, and the reliability of one-shot NAS remains largely unexplored in time-series domains like Human Activity Recognition (HAR).

This thesis presents an implementation of the Single Path One-Shot (SPOS) algorithm that is compatible with the Optuna framework and uses the elasticAI.explorer search space definition. The approach is evaluated against a non-weight-sharing baseline NAS on the PAMAP2 dataset for HAR under equivalent computational budgets.

Both approaches find architectures with comparable test accuracies of 86.5%. While SPOS accelerates search by 1.7x, it yields models that are 3.6x slower and 4.3x larger, indicating a tendency to converge on large and complex architectures. Enforcing a < 10M FLOP budget cuts FLOPs but nearly doubles search cost, does not mitigate the latency issue and hurts accuracy by a small amount. Lastly, Spearman's rho=0.25 and Kendall's tau=0.18 indicate supernet validation correlates weakly with standalone performance, barely exceeding random chance.

The results indicate that supernet validation is an imperfect proxy for architecture quality. Furthermore, this thesis was not able to dispute the criticism that the success of one-shot, and weight-sharing NAS in general often stems from an overly constrained search space rather than the search algorithm itself.
