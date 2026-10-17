from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.utils.gp_tfbind8 import ADNGaussianModel
from experiments.utils.gp_gradient_ascent import gp_gradient_ascent
from experiments.utils.utils import str_to_discrete, discrete_to_str, discrete_to_logits, logits_to_discrete
import numpy as np
import torch
import random
import gpytorch
import argparse
import re
import os
import design_bench

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(CURRENT_DIR)  # experiments/
GP_DATA_PATH = os.path.join(EXPERIMENTS_DIR, "checkpoints", "training_data.pth")
GP_PARAMS_PATH = os.path.join(EXPERIMENTS_DIR, "checkpoints", "model_state.pth")
LM_NAME = "Qwen/Qwen2.5-7B-Instruct"

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

# LM call support functions
TASK_DESCRIPTION = '''
    You aim to find a new length-8 DNA sequence, composed of A, C, G, and T,
    with maximum binding affinity with the transcription factor SIX6_REF_R1.
'''

INSTRUCTION = '''
    Given the new candidate sequences, rank them in order of predicted binding affinity from highest to lowest.
    Output only the rankings the new sequences, one sequence per line, no numbering, no explanation.
'''

def format_prompt(samples, cands_str):
    '''
    Format the prompt with task description, instruction, in-context samples, and candidate sequences.
    '''
    in_context_samples = "\n".join(
        f"DNA Sequence: {seq}, Binding Affinity: {score:.2f}"
        for seq, score in samples
    )

    candidates = "\n".join(
        f"DNA Sequence: {seq}"
        for seq in cands_str
    )

    return (
        f"Task Description:\n{TASK_DESCRIPTION}\n\n"
        f"Offline Dataset:\n{in_context_samples}\n\n"
        f"Instruction:\n{INSTRUCTION}\n\n"
        f"{candidates}"
    )

def get_response(lm, tokenizer, prompt):
    '''
    Get response from lm.
    '''
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(lm.device)
    
    with torch.no_grad():
        outputs = lm.generate(**inputs, max_new_tokens=256)
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# parse designs' ranking from the response 
def parse_ranking(response, cands_str):
    """Read ranked sequences from text; drop invalid/duplicates, append any missing."""
    valid, seen, ranking = set(cands_str), set(), []
    for s in re.findall(r"[ACGT]{8}", response.upper()):
        if s in valid and s not in seen:
            ranking.append(s)
            seen.add(s)
    return ranking + [s for s in cands_str if s not in seen]

# Evaluate function using AUPCC
def aupcc(true_ranking, pred_ranking):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gp-data-path", type=str, default=GP_DATA_PATH)
    parser.add_argument("--gp-params-path", type=str, default=GP_PARAMS_PATH)
    parser.add_argument("--grad-lr", type=float, default=0.1)
    parser.add_argument("--grad-num-steps", type=int, default=100)
    parser.add_argument("--n-sample", type=int, default=16)
    parser.add_argument("--lm-name", type=str, default=LM_NAME)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tfbind8 task and train data into gp
    task = design_bench.make("TFBind8-Exact-v0")
    X = task.x
    y = task.y

    gp_train_data = prepare_data(task, args.gp_data_path)
    X_train, y_train = gp_train_data["X_train"], gp_train_data["y_train"]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    gp_model = ADNGaussianModel(X_train, y_train, likelihood)
    state_dict = torch.load(args.gp_params_path, map_location=device, weights_only=False)
    gp_model.load_state_dict(state_dict)

    likelihood.eval()
    gp_model.eval()
    
    # Get random in-context samples 
    X_sample, y_sample = get_random_samples(args.n_sample, X, y)

    samples_str = discrete_to_str(X_sample)
    samples_score = y_sample.squeeze().tolist()  # ensure flat scalars, not lists
    samples = [(samples_str[i], samples_score[i]) for i in range(args.n_sample)]

    # normalize & gradient ascent
    mean_X, std_X = gp_train_data["mean_X"], gp_train_data["std_X"]
    sample_logits = discrete_to_logits(X_sample, task)
    sample_logits = sample_logits.reshape((len(sample_logits), -1))  # (n, 24)
    sample_logits = (sample_logits - mean_X) / std_X

    cands_logits = gp_gradient_ascent(gp_model, sample_logits, args.grad_lr, args.grad_num_steps, optimizer="adam")
    cands_logits = cands_logits * std_X + mean_X
    cands_logits = cands_logits.reshape((len(cands_logits), 8, -1))  # (n, 8, 3)
    cands_discrete = logits_to_discrete(cands_logits)
    cands_str = discrete_to_str(cands_discrete)

    # Deduplicate candidates (gradient ascent can converge to same sequence)
    seen = set()
    unique_idx = []
    for i, s in enumerate(cands_str):
        if s not in seen:
            seen.add(s)
            unique_idx.append(i)
    cands_str = [cands_str[i] for i in unique_idx]
    cands_discrete = cands_discrete[unique_idx]
    
    # ranking llm
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    lm = AutoModelForCausalLM.from_pretrained(
        args.lm_name,
        torch_dtype=torch.float16,
        device_map=device
    )

    prompt = format_prompt(samples, cands_str)
    response = get_response(lm, tokenizer, prompt)
    pred_ranking = parse_ranking(response, cands_str)

    # get the true ranking use the oracle function
    true_score = task.predict(cands_discrete).ravel()
    order = np.argsort(-true_score, kind="stable")      # best first
    true_ranking = [cands_str[i] for i in order]

    aupcc_score = aupcc(true_ranking, pred_ranking)
    print(aupcc_score, true_ranking, pred_ranking)

if __name__ == "__main__":
    main()