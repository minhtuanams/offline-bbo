import torch
import torch.nn as nn
import gpytorch

def gp_gradient_ascent(model, x_init, lr=0.05, num_steps=100, optimizer="adam"):
    x_opt = x_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x_opt], lr=lr) if optimizer == "adam" else torch.optim.SGD([x_opt], lr=lr)

    # Ascent loop
    for step in range(num_steps):
        opt.zero_grad()
        output = model(x_opt)
        score = output.mean

        loss = -torch.mean(score)
        loss.backward()
        opt.step()

    final_x = x_opt.detach()

    return final_x
