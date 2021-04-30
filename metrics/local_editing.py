"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from torchvision import utils, transforms
from torch.nn import functional as F
from training.dataset import DataSetTestLocalEditing
from torch.utils import data
import random
import time

from tqdm import tqdm
import os

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--data_dir")

    args = parser.parse_args()

    size = 256
    channel_multiplier = 2

    batch_size = args.batch

    if "celeba_hq" in args.data_dir:
        parts = os.listdir(args.data_dir)
    elif "afhq" in args.data_dir:
        parts = [""]

    print("parts", parts)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    mse_loss = nn.MSELoss(reduction="none")

    inMSE_all_part = []
    outMSE_all_part = []

    with torch.no_grad():
        for part in parts:
            print(part)
            inMSE_ref_list = []
            outMSE_src_list = []

            dataset = DataSetTestLocalEditing(
                os.path.join(args.data_dir, part), transform
            )
            assert len(dataset) == 500 / 2

            loader = data.DataLoader(
                dataset,
                batch_size,
                sampler=data_sampler(dataset, shuffle=False),
                num_workers=args.num_workers,
                pin_memory=True,
            )

            for mask, image_reference, image_source, image_synthesized in tqdm(
                loader, mininterval=1
            ):
                N = len(mask)

                mask, image_reference, image_source, image_synthesized = (
                    mask.to(device),
                    image_reference.to(device),
                    image_source.to(device),
                    image_synthesized.to(device),
                )

                MSE_between_src = mse_loss(image_synthesized, image_source)
                MSE_between_ref = mse_loss(image_synthesized, image_reference)

                inMSE_mask_count = (mask == 1).sum()
                outMSE_mask_count = (mask == -1).sum()

                if inMSE_mask_count == 0:
                    # print("no mask is found")
                    continue

                assert inMSE_mask_count + outMSE_mask_count == size * size * N * 3

                dummy = torch.zeros(MSE_between_src.shape, device=device)

                inMSE_ref = torch.where(mask == 1, MSE_between_ref, dummy)
                inMSE_ref = inMSE_ref.sum() / inMSE_mask_count

                outMSE_src = torch.where(mask == -1, MSE_between_src, dummy)
                outMSE_src = outMSE_src.sum() / outMSE_mask_count
                inMSE_ref_list.append(inMSE_ref.mean())
                outMSE_src_list.append(outMSE_src.mean())

            inMSE_ref = sum(inMSE_ref_list) / len(inMSE_ref_list)
            outMSE_src = sum(outMSE_src_list) / len(outMSE_src_list)

            inMSE_all_part.append(inMSE_ref)
            outMSE_all_part.append(outMSE_src)

            print(f"{inMSE_ref:.3f}, {outMSE_src:.3f}")

        print(f"average in inMSE_ref, outMSE_src")
        print(
            f"{sum(inMSE_all_part) / len(inMSE_all_part):.3f}, {sum(outMSE_all_part) / len(outMSE_all_part):.3f}"
        )
