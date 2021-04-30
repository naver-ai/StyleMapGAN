"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import argparse
import os
from torchvision import transforms
import training.lpips as lpips
import torch.nn as nn
from PIL import Image
import torch

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_folder_path", type=str, default="expr/reconstruction/celeba_hq"
    )
    args = parser.parse_args()
    image_folder_path = args.image_folder_path

    # images(0~1) are converted to -1 ~ 1
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    mse_loss = nn.MSELoss(size_average=True)
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    fake_filenames = []
    real_filenames = []

    print(image_folder_path)
    dataset_len = 500

    for i in range(dataset_len):
        fake_filename = os.path.join(image_folder_path, f"{i}_recon.png")
        real_filename = os.path.join(image_folder_path, f"{i}_real.png")

        if os.path.isfile(fake_filename) and os.path.isfile(real_filename):
            fake_filenames.append(fake_filename)
            real_filenames.append(real_filename)
        else:
            print(f"{fake_filename} or {real_filename} doesn't exists")
            break

    print(len(fake_filenames), len(real_filenames))
    assert len(fake_filenames) == dataset_len
    assert len(real_filenames) == dataset_len

    mse_results = []
    lpips_results = []

    with torch.no_grad():
        for fake_filename, real_filename in zip(fake_filenames, real_filenames):
            fake_img = transform(Image.open(fake_filename).convert("RGB")).to(device)
            real_img = transform(Image.open(real_filename).convert("RGB")).to(device)
            # assert real_img.shape == (3, 256, 256)

            mse = mse_loss(fake_img, real_img).item()
            lpips = percept(fake_img, real_img).item()

            mse_results.append(mse)
            lpips_results.append(lpips)

    mse_mean = sum(mse_results) / len(mse_results)
    lpips_mean = sum(lpips_results) / len(lpips_results)

    print(mse_mean, lpips_mean)
