import numpy as np
import torch


class BaseSampler:
    def __init__(self, X, y, n_samples, seed=None):
        self.X = X                       # (N, 8) discrete designs
        self.y = np.asarray(y).ravel()   # (N,) scores
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def sample(self, ref=None):
        '''Return (X_sample, y_sample): (n, 8) discrete designs and (n,) scores.'''
        raise NotImplementedError

    def random_samples(self):
        idx = self.rng.choice(len(self.X), size=self.n_samples, replace=False)
        return self.X[idx], self.y[idx]


class ConstantSampler(BaseSampler):
    '''Return the fixed examples supplied at initialization.'''
    def __init__(self, X, y, idx):
        self.X = X
        self.y = np.asarray(y).ravel()  # One scalar score per design, as in BaseSampler.
        self.idx = idx

    def sample(self, ref=None):
        return self.X[self.idx], self.y[self.idx]


class RandomSampler(BaseSampler):
    '''New random samples on every call.'''

    def sample(self, ref=None):
        return self.random_samples()


class SmartSampler(BaseSampler):
    '''The n_samples designs closest (Hamming distance) to ref.'''

    def __init__(self, X, y, n_samples, device=None, seed=None):
        super().__init__(X, y, n_samples, seed)
        self.device = device
        self.X_t = torch.as_tensor(X, device=device)

    def sample(self, ref):
        ref = torch.as_tensor(np.atleast_2d(ref), device=self.device)       # (m, 8)
        dist = (self.X_t[:, None, :] != ref[None, :, :]).sum(-1).min(dim=1)  # nearest ref
        dist = dist.values.cpu().numpy().astype(float)
        dist[dist == 0] = np.inf  # never return the reference itself

        # Shuffle first so ties (Hamming only has 9 values) are broken randomly
        order = self.rng.permutation(len(dist))
        idx = order[np.argsort(dist[order], kind="stable")][: self.n_samples]
        return self.X[idx], self.y[idx]
