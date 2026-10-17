"""
Support functions to convert DNA sequence.
str (string) <-> discrete (NumPy array) <-> logits (PyTorch tensor)
"""

import torch
import numpy as np


def str_to_discrete(seqs_str):
    nt_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    seqs_list = [[nt_idx[nt] for nt in seq] for seq in seqs_str]
    return np.array(seqs_list, dtype=np.int32)


def discrete_to_str(seqs_discrete):
    idx_nt = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    return [
        "".join(idx_nt[int(idx)] for idx in seq)
        for seq in seqs_discrete
    ]


def discrete_to_logits(seqs_discrete, task, device=None, dtype=torch.float32):
    logits = torch.as_tensor(np.asarray(task.to_logits(seqs_discrete)), dtype=dtype)
    return logits.to(device) if device is not None else logits


def logits_to_discrete(seqs_logits):
    seqs_logits_np = seqs_logits.detach().cpu().numpy()

    # Assumes 3 logits represent C, G, T, with A as the zero-logit baseline.
    zeros_np = np.zeros(
        (*seqs_logits_np.shape[:-1], 1),
        dtype=seqs_logits_np.dtype
    )

    seqs_logits_full = np.concatenate(
        [zeros_np, seqs_logits_np],
        axis=-1
    )

    return np.argmax(seqs_logits_full, axis=-1).astype(np.int32)