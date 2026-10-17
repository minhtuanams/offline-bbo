import os
import torch
import gpytorch
from torch import nn
from src.data.load_tfbind8 import load_tfbind8
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(CURRENT_DIR)  # experiments/
CHECKPOINTS_DIR = os.path.join(EXPERIMENTS_DIR, "checkpoints")

class ADNGaussianModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def prepare_data(n_train=10000):
    # Prepare fitting data
    task, X, y = load_tfbind8()
    train_indices = torch.randperm(len(X))[:n_train]
    indices_tensor = torch.tensor([False]*len(X), dtype=torch.bool)
    indices_tensor[train_indices] = True

    X_train = X[train_indices]
    y_train = y[train_indices]

    # Save X_train, y_train
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    torch.save(
        {
            "X_train": X_train,
            "y_train": y_train,
        },
        os.path.join(CHECKPOINTS_DIR, 'training_data.pth')
    )

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

    return X_train, y_train


def main():
    X_train, y_train = prepare_data()
    
    #GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ADNGaussianModel(X_train, y_train, likelihood)

    # Fitting
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.2) # Includes GuassianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    
    # Save training data / parameters
    torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, 'model_state.pth'))

if __name__ == "__main__":
    main()