# -*- coding: utf-8 -*-
"""diffusion_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL

# A Diffusion Model from Scratch in Pytorch

In this notebook I want to build a very simple (as few code as possible) Diffusion Model for generating car images. I will explain all the theoretical details in the YouTube video.


**Sources:**
- Github implementation [Denoising Diffusion Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- Niels Rogge, Kashif Rasul, [Huggingface notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=3a159023)
- Papers on Diffusion models ([Dhariwal, Nichol, 2021], [Ho et al., 2020] ect.)

## Investigating the dataset

As dataset we use the StandordCars Dataset, which consists of around 8000 images in the train set. Let's see if this is enough to get good results ;-)


Later in this notebook we will do some additional modifications to this dataset, for example make the images smaller, convert them to tensors ect.

# Building the Diffusion Model

## Step 1: The forward process = Noise scheduler

We first need to build the inputs for our model, which are more and more noisy images. Instead of doing this sequentially, we can use the closed form provided in the papers to calculate the image for any of the timesteps individually.

**Key Takeaways**:
- The noise-levels/variances can be pre-computed
- There are different types of variance schedules
- We can sample each timestep image independently (Sums of Gaussians is also Gaussian)
- No model is needed in this forward step

**Further improvements that can be implemented:**
- Residual connections
- Different activation functions like SiLU, GWLU, ...
- BatchNormalization
- GroupNormalization
- Attention
- ...

## Step 2: The backward process = U-Net

For a great introduction to UNets, have a look at this post: https://amaarora.github.io/2020/09/13/unet.html.


**Key Takeaways**:
- We use a simple form of a UNet for to predict the noise in the image
- The input is a noisy image, the ouput the noise in the image
- Because the parameters are shared accross time, we need to tell the network in which timestep we are
- The Timestep is encoded by the transformer Sinusoidal Embedding
- We output one single value (mean), because the variance is fixed


## Step 3: The loss

**Key Takeaways:**
- After some maths we end up with a very simple loss function
- There are other possible choices like L2 loss ect.

## Sampling
- Without adding @torch.no_grad() we quickly run out of memory, because pytorch tacks all the previous images for gradient calculation
- Because we pre-calculated the noise variances for the forward pass, we also have to use them when we sequentially perform the backward process


"""

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import diffusion_config
from diffusion_model import SimpleUnet, get_loss
from utils import (
    StanfordCars,
    load_transformed_dataset,
    plot_diffusion,
    plot_denoising,
    show_images,
)

data = StanfordCars(root_path="../data/stanford_cars/cars_train/cars_train")
show_images(data)

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=diffusion_config.BATCH_SIZE, shuffle=True, drop_last=True)

plot_diffusion(dataloader)

model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
model

# Training
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(diffusion_config.NUM_EPOCHS):
    for step, batch in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        t = torch.randint(0, diffusion_config.MAX_TIMESTEP, (diffusion_config.BATCH_SIZE,), device=device).long()

        loss = get_loss(model, batch, t, device)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            plot_denoising(model, device)

    torch.save(model.state_dict(), "models/car_model.pth")

plt.show()
