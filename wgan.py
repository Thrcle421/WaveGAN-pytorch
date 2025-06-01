import logging
import torch
from torch import autograd
from torch import optim
from utils import save_samples
import numpy as np
import pprint
import argparse
import os


LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)


def compute_discr_loss_terms(model_dis, model_gen, real_data_v, noise_v,
                             batch_size, latent_dim,
                             lmbda, use_cuda, compute_grads=False):
    # Reset gradients
    model_dis.zero_grad()

    # a) Compute loss contribution from real training data and backprop
    # (negative of the empirical mean, w.r.t. the data distribution, of the discr. output)
    D_real = model_dis(real_data_v)
    D_real = D_real.mean()

    # b) Compute loss contribution from generated data and backprop
    # (empirical mean, w.r.t. the generator distribution, of the discr. output)
    # Generate noise in latent space

    # Generate data by passing noise through the generator
    with torch.no_grad():
        fake = model_gen(noise_v)
    fake = autograd.Variable(fake.data, requires_grad=True)
    inputv = fake
    D_fake = model_dis(inputv)
    D_fake = D_fake.mean()

    # c) Compute gradient penalty and backprop
    gradient_penalty = calc_gradient_penalty(model_dis, real_data_v.data,
                                             fake.data,
                                             batch_size, lmbda,
                                             use_cuda=use_cuda)

    # Compute the total discriminator loss: maximize D_real - D_fake - gradient_penalty
    # Which is equivalent to minimizing -(D_real - D_fake) + gradient_penalty
    # So we minimize: D_fake - D_real + gradient_penalty
    D_cost = D_fake - D_real + gradient_penalty

    if compute_grads:
        D_cost.backward()

    # Compute metrics and record in batch history
    Wasserstein_D = D_real - D_fake

    return D_cost, Wasserstein_D


def compute_gener_loss_terms(model_dis, model_gen, batch_size, latent_dim,
                             use_cuda, compute_grads=False):
    # Reset generator gradients
    model_gen.zero_grad()

    # Sample from the generator
    noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
    if use_cuda:
        noise = noise.cuda()
    noise_v = autograd.Variable(noise)
    fake = model_gen(noise_v)

    # Compute generator loss and backprop
    # We want to maximize D(G(z)), which is equivalent to minimizing -D(G(z))
    G = model_dis(fake)
    G = G.mean()
    G_cost = -G  # Negative because we want to minimize this

    if compute_grads:
        G_cost.backward()

    return G_cost


def np_to_input_var(data, use_cuda):
    data = data[:, np.newaxis, :]
    data = torch.Tensor(data)
    if use_cuda:
        data = data.cuda()
    return autograd.Variable(data)


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def calc_gradient_penalty(model_dis, real_data, fake_data, batch_size, lmbda, use_cuda=True):
    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    # Interpolate between real and fake data
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = model_dis(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to encourage discriminator
    # to be a 1-Lipschitz function
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda
    return gradient_penalty


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def train_wgan(model_gen, model_dis, train_gen, valid_data, test_data,
               num_epochs, batches_per_epoch, batch_size, output_dir=None,
               lmbda=0.1, use_cuda=True, discriminator_updates=5, epochs_per_sample=10,
               sample_size=20, lr=1e-4, beta_1=0.5, beta_2=0.9, latent_dim=100):

    # 设置设备和显示GPU信息
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        LOGGER.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        # 在训练前先清理缓存
        torch.cuda.empty_cache()
        # 初始记录内存状态
        LOGGER.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        LOGGER.info(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        LOGGER.info("Using CPU")
    
    # 移动模型到适当的设备
    model_gen = model_gen.to(device)
    model_dis = model_dis.to(device)
    
    # 使用DataParallel进行多GPU训练（如果可用）
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        LOGGER.info(f"Using {torch.cuda.device_count()} GPUs!")
        model_gen = torch.nn.DataParallel(model_gen)
        model_dis = torch.nn.DataParallel(model_dis)

    # Initialize optimizers for each model
    optimizer_gen = optim.Adam(model_gen.parameters(), lr=lr,
                               betas=(beta_1, beta_2))
    optimizer_dis = optim.Adam(model_dis.parameters(), lr=lr,
                               betas=(beta_1, beta_2))

    # Sample noise used for seeing the evolution of generated output samples throughout training
    sample_noise = torch.Tensor(sample_size, latent_dim).uniform_(-1, 1)
    sample_noise = sample_noise.to(device)
    sample_noise_v = autograd.Variable(sample_noise)

    samples = {}
    history = []

    train_iter = iter(train_gen)
    valid_data_v = np_to_input_var(valid_data['X'], use_cuda)
    test_data_v = np_to_input_var(test_data['X'], use_cuda)

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        LOGGER.info(f"Epoch: {epoch + 1}/{num_epochs}")

        epoch_history = []
        
        # 记录每个epoch的GPU内存使用情况
        if device.type == 'cuda':
            LOGGER.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            LOGGER.info(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

        for batch_idx in range(batches_per_epoch):

            # Set model parameters to require gradients to be computed and stored
            for p in model_dis.parameters():
                p.requires_grad = True

            # Initialize the metrics for this batch
            batch_history = {
                'discriminator': [],
                'generator': {}
            }

            # Discriminator Training Phase:
            # -> Train discriminator k times
            for iter_d in range(discriminator_updates):
                # Get real examples
                real_data_v = np_to_input_var(next(train_iter)['X'], use_cuda)

                # Get noise
                noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
                noise = noise.to(device)
                noise_v = autograd.Variable(noise)

                # Get new batch of real training data
                D_cost_train, D_wass_train = compute_discr_loss_terms(
                    model_dis, model_gen, real_data_v, noise_v, batch_size,
                    latent_dim,
                    lmbda, use_cuda, compute_grads=True)

                # Update the discriminator
                optimizer_dis.step()

                D_cost_valid, D_wass_valid = compute_discr_loss_terms(
                    model_dis, model_gen, valid_data_v, noise_v, batch_size,
                    latent_dim,
                    lmbda, use_cuda, compute_grads=False)

                # Move tensors to CPU for logging
                D_cost_train_cpu = D_cost_train.cpu()
                D_cost_valid_cpu = D_cost_valid.cpu()
                D_wass_train_cpu = D_wass_train.cpu()
                D_wass_valid_cpu = D_wass_valid.cpu()

                batch_history['discriminator'].append({
                    'cost': D_cost_train_cpu.item(),
                    'wasserstein_cost': D_wass_train_cpu.item(),
                    'cost_validation': D_cost_valid_cpu.item(),
                    'wasserstein_cost_validation': D_wass_valid_cpu.item()
                })

            ############################
            # (2) Update G network
            ###########################

            # Freeze the discriminator
            for p in model_dis.parameters():
                p.requires_grad = False  # to avoid computation

            # Train the generator
            G_cost = compute_gener_loss_terms(model_dis, model_gen, batch_size,
                                              latent_dim, use_cuda,
                                              compute_grads=True)

            # Update the generator
            optimizer_gen.step()

            # Move to CPU for logging
            G_cost_cpu = G_cost.cpu()

            batch_history['generator'] = {
                'cost': G_cost_cpu.item()
            }

            epoch_history.append(batch_history)

        history.append(epoch_history)

        # Sample from generator
        if epoch % epochs_per_sample == 0:
            model_gen.eval()
            # Generate samples for inspection
            with torch.no_grad():
                sample_out = model_gen(sample_noise_v)
            samples[epoch] = sample_out.cpu().numpy()

            if output_dir is not None:
                save_samples(samples[epoch], epoch, output_dir)

            # set model back to training
            model_gen.train()

    LOGGER.info("Training complete")
    
    # 清理缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return model_gen, model_dis, history, {
        'valid_wasserstein_cost': D_wass_valid_cpu.item(),
        'valid_cost': D_cost_valid_cpu.item()
    }, samples
