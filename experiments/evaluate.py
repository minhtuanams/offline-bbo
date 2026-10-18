from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.utils.gp_tfbind8 import ADNGaussianModel
from experiments.utils.gp_gradient_ascent import gp_gradient_ascent
from experiments.utils.utils import str_to_discrete, discrete_to_str, discrete_to_logits, logits_to_discrete
from experiments.utils.lm_model import LanguageModel
from experiments.rankers.full_ranker import FullRanker
from experiments.rankers.pairwise_ranker import PairwiseRanker

from tqdm import tqdm
import numpy as np
import torch
import random
import gpytorch
import argparse
import re
import os
import design_bench
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # experiments/
GP_DATA_PATH = os.path.join(CURRENT_DIR, "checkpoints", "training_data.pth")
GP_PARAMS_PATH = os.path.join(CURRENT_DIR, "checkpoints", "model_state.pth")
SAVED_PATH = os.path.join(CURRENT_DIR, "outputs", "ranker_auscc_report_v2.json")
LM_NAME = "qwen2.5-7b"

def prepare_data(task, gp_data_path):
    '''
    Prepare data for gp_model.
    '''
    # Prepare fitting data
    train_data = torch.load(gp_data_path, weights_only=False)
    X_train = train_data["X_train"]
    y_train = train_data["y_train"]

    # Convert to continuous logits space
    X_logits = task.to_logits(X_train)
    X_logits = X_logits.reshape((len(X_logits), -1))
    X_logits = torch.from_numpy(X_logits)
    X_train = X_logits

    y_train = torch.from_numpy(y_train)
    y_train = y_train.squeeze()

    # Normalize
    mean_X = torch.mean(X_train, dim=0)
    std_X = torch.std(X_train, dim=0)
    X_train = (X_train - mean_X) / std_X

    mean_y = torch.mean(y_train, dim=0)
    std_y = torch.std(y_train, dim=0)
    y_train = (y_train - mean_y) / std_y

    return {
        "X_train": X_train,
        "y_train": y_train,
        "mean_X": mean_X,
        "std_X": std_X,
        "mean_y": mean_y,
        "std_y": std_y,
    }

def get_random_samples(n_sample, X, y):
    '''
    Get random n_sample samples.
    '''
    sample_indices = torch.randperm(len(X))[:n_sample]
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]

    return X_sample, y_sample

# Evaluate function using AUPCC
def cal_aupcc_score(true_ranking, pred_ranking):
    """Area under the precision-coverage curve, in [0, 1].

    Inputs: two ranked lists of the same items, best first.
    """
    n = len(true_ranking)
    if n == 0 or len(set(true_ranking)) != n or set(true_ranking) != set(pred_ranking):
        raise ValueError("Inputs must be non-empty, duplicate-free, with identical items.")

    true_seen, pred_seen = set(), set()
    overlap, total = 0, 0.0
    for k, (t, p) in enumerate(zip(true_ranking, pred_ranking), start=1):
        # Incremental top-k intersection size
        overlap += 1 if t == p else (t in pred_seen) + (p in true_seen)
        true_seen.add(t)
        pred_seen.add(p)
        total += overlap / k  # precision@k

    return total / n

def evaluate(task, X, y, gp_model, gp_train_data, rankers, args):
    '''One trial: sample -> gradient ascent -> candidates -> rank -> AUPCC per ranker.'''
    mean_X, std_X = gp_train_data["mean_X"], gp_train_data["std_X"]

    # In-context samples
    X_sample, y_sample = get_random_samples(args.n_sample, X, y)
    samples = list(zip(discrete_to_str(X_sample), y_sample.squeeze().tolist()))

    # Candidates from gradient ascent
    sample_logits = discrete_to_logits(X_sample, task)
    sample_logits = sample_logits.reshape((len(sample_logits), -1))
    sample_logits = (sample_logits - mean_X) / std_X

    cands_logits = gp_gradient_ascent(
        gp_model, sample_logits, args.grad_lr, args.grad_num_steps, optimizer="adam"
    )
    cands_logits = (cands_logits * std_X + mean_X).reshape((len(cands_logits), 8, -1))
    cands_discrete = logits_to_discrete(cands_logits)
    cands_str = discrete_to_str(cands_discrete)

    # Deduplicate (gradient ascent can converge to the same sequence)
    unique_idx = list({s: i for i, s in reversed(list(enumerate(cands_str)))}.values())
    unique_idx.sort()
    cands_str = [cands_str[i] for i in unique_idx]
    cands_discrete = cands_discrete[unique_idx]

    # True ranking from the oracle
    true_score = task.predict(cands_discrete).ravel()
    true_ranking = [cands_str[i] for i in np.argsort(-true_score, kind="stable")]

    # Score every ranker on the same candidates
    scores = {}
    for name, ranker in rankers.items():
        pred_ranking = ranker.rank(samples, cands_str)
        scores[name] = cal_aupcc_score(true_ranking, pred_ranking)
    return scores, len(cands_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gp-data-path", type=str, default=GP_DATA_PATH)
    parser.add_argument("--gp-params-path", type=str, default=GP_PARAMS_PATH)
    parser.add_argument("--saved-path", type=str, default=SAVED_PATH)
    parser.add_argument("--grad-lr", type=float, default=0.1)
    parser.add_argument("--grad-num-steps", type=int, default=100)
    parser.add_argument("--n-sample", type=int, default=32)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--lm-name", type=str, default=LM_NAME)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load once: task, GP, LM, rankers
    task = design_bench.make("TFBind8-Exact-v0")
    X, y = task.x, task.y

    gp_train_data = prepare_data(task, args.gp_data_path)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = ADNGaussianModel(gp_train_data["X_train"], gp_train_data["y_train"], likelihood)
    gp_model.load_state_dict(torch.load(args.gp_params_path, map_location=device, weights_only=False))
    likelihood.eval()
    gp_model.eval()

    lm = LanguageModel(args.lm_name)
    rankers = {"full": FullRanker(lm), "pairwise": PairwiseRanker(lm)}

    # Repeat trials and collect results
    trials = []
    for t in tqdm(range(args.n_trials)):
        scores, n_cands = evaluate(task, X, y, gp_model, gp_train_data, rankers, args)
        trials.append({"trial": t + 1, "n_cands": n_cands, "aupcc": scores})

    report = {
        "config": {
            "lm_name": args.lm_name,
            "n_sample": args.n_sample,          # in-context samples per trial
            "n_trials": args.n_trials,
            "grad_lr": args.grad_lr,
            "grad_num_steps": args.grad_num_steps,
        },
        "avg_n_cands": float(np.mean([t["n_cands"] for t in trials])),  # unique candidates fed to rankers
        "aupcc": {
            name: {
                "mean": float(np.mean([t["aupcc"][name] for t in trials])),
                "std": float(np.std([t["aupcc"][name] for t in trials])),
            }
            for name in rankers
        },
        "trials": trials,
    }

    with open(args.saved_path, "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()