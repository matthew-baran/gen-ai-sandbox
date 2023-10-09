import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import math
import os

from diffusion_model import sample_timestep, forward_diffusion_sample
from constants import T

IMG_SIZE = 64


class StanfordCars(torch.utils.data.Dataset):
    def __init__(self, root_path, transform=None):
        self.images = [os.path.join(root_path, file) for file in os.listdir(root_path)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]
        image = Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = StanfordCars(
        root_path="../data/stanford_cars/cars_train/cars_train",
        transform=data_transform,
    )

    test = StanfordCars(
        root_path="../data/stanford_cars/cars_test/cars_test", transform=data_transform
    )
    return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


@torch.no_grad()
def sample_plot_image(model, device="cpu"):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Current model de-noising for decreasing time t", y=1.08)
    num_images = 9
    subplot_dim = math.ceil(math.sqrt(num_images))
    plot_steps = np.floor(np.linspace(0, T - 1, num_images))

    subplot_idx = 0
    for time_val in range(0, T)[::-1]:
        t = torch.full((1,), time_val, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

        if time_val in plot_steps:
            ax = plt.subplot(subplot_dim, subplot_dim, subplot_idx + 1)
            ax.set_title(f"t={time_val}")
            show_tensor_image(img.detach().cpu())
            subplot_idx += 1
    plt.show(block=False)


def show_images(dataset, num_samples=20, cols=4):
    """Plots some samples from the dataset"""
    plt.figure(figsize=(15, 15))
    plt.title(f"First {num_samples} images in the dataset")
    plt.axis("off")
    subplot_dim = math.ceil(math.sqrt(num_samples))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(subplot_dim, subplot_dim, i + 1)
        plt.imshow(img)

def plot_diffusion(dataloader):
    # Simulate forward diffusion
    image = next(iter(dataloader))[0]

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Noise process for increasing time t", y=1.08)
    num_images = 9
    subplot_dim = math.ceil(math.sqrt(num_images))

    for idx, time_val in enumerate(np.floor(np.linspace(0, T - 1, num_images))):
        t = torch.Tensor([time_val]).type(torch.int64)
        ax = plt.subplot(subplot_dim, subplot_dim, idx + 1)
        ax.set_title(f"t={time_val}")
        img, _ = forward_diffusion_sample(image, t)
        show_tensor_image(img)