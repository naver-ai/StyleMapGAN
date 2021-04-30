"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from io import BytesIO
import sys
from os import path
import lmdb
from PIL import Image
import argparse
import numpy as np
import os
import pickle
import itertools
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils import data
from torchvision import transforms, utils
from torchvision.utils import save_image


sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from training.dataset import GTMaskDataset


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def group_pair_GT():
    device = "cuda"
    args.n_sample = 500

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    images_size = 256  # you can use other resolution for calculating if your LMDB(args.path) has different resolution.
    dataset = GTMaskDataset(args.path, transform, images_size)

    parts_index = {
        "all": None,
        "background": [0],
        "skin": [1],
        "eyebrow": [6, 7],
        "eye": [3, 4, 5],
        "ear": [8, 9, 15],
        "nose": [2],
        "lip": [10, 11, 12],
        "neck": [16, 17],
        "cloth": [18],
        "hair": [13, 14],
    }

    indexes = range(args.n_sample)

    similarity_dict = {}
    parts = parts_index.keys()

    for part in parts:
        similarity_dict[part] = {}

    for src, ref in tqdm(
        itertools.combinations(indexes, 2),
        total=sum(1 for _ in itertools.combinations(indexes, 2)),
    ):
        _, mask1 = dataset[src]
        _, mask2 = dataset[ref]
        mask1 = mask1.to(device)
        mask2 = mask2.to(device)
        for part in parts:
            if part == "all":
                similarity = torch.sum(mask1 == mask2).item() / (images_size ** 2)
                similarity_dict["all"][src, ref] = similarity
            else:
                part1 = torch.zeros(
                    [images_size, images_size], dtype=torch.bool, device=device
                )
                part2 = torch.zeros(
                    [images_size, images_size], dtype=torch.bool, device=device
                )

                for p in parts_index[part]:
                    part1 = part1 | (mask1 == p)
                    part2 = part2 | (mask2 == p)

                intersection = (part1 & part2).sum().float().item()
                union = (part1 | part2).sum().float().item()
                if union == 0:
                    similarity_dict[part][src, ref] = 0.0
                else:
                    sim = intersection / union
                    similarity_dict[part][src, ref] = sim

    sorted_similarity = {}

    for part, similarities in similarity_dict.items():
        all_indexes = set(range(args.n_sample))
        sorted_similarity[part] = []

        sorted_list = sorted(similarities.items(), key=(lambda x: x[1]), reverse=True)

        for (i1, i2), prob in sorted_list:
            if (i1 in all_indexes) and (i2 in all_indexes):
                all_indexes -= {i1, i2}
                sorted_similarity[part].append(((i1, i2), prob))
            elif len(all_indexes) == 0:
                break

        assert len(sorted_similarity[part]) == args.n_sample // 2

    with open(
        f"{args.save_dir}/{args.dataset_name}_test_{args.mask_origin}_sorted_pair.pkl",
        "wb",
    ) as handle:
        pickle.dump(sorted_similarity, handle)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--save_dir", type=str, default="../data/celeba_hq/local_editing"
    )

    args = parser.parse_args()
    args.dataset_name = "celeba_hq"
    os.makedirs(args.save_dir, exist_ok=True)
    args.path = f"../data/{args.dataset_name}"

    with torch.no_grad():
        # our CelebA-HQ test dataset contains 500 images
        # change this value if you have the different number of GT_labels
        args.n_sample = 500
        group_pair_GT()
