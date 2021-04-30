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
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils, transforms
import numpy as np
from torchvision.datasets import ImageFolder
from training.dataset import (
    MultiResolutionDataset,
    GTMaskDataset,
)
from scipy import linalg
import random
import time
import os
from tqdm import tqdm
from copy import deepcopy
import cv2
from PIL import Image
from itertools import combinations
from training.model import Generator, Encoder

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def save_image(img, path, normalize=True, range=(-1, 1)):
    utils.save_image(
        img,
        path,
        normalize=normalize,
        range=range,
    )


def save_images(imgs, paths, normalize=True, range=(-1, 1)):
    for img, path in zip(imgs, paths):
        save_image(img, path, normalize=normalize, range=range)


def make_noise(batch, latent_channel_size, device):
    return torch.randn(batch, latent_channel_size, device=device)


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


class Model(nn.Module):
    def __init__(self, device="cuda"):
        super(Model, self).__init__()
        self.g_ema = Generator(
            args.size,
            args.mapping_layer_num,
            args.latent_channel_size,
            args.latent_spatial_size,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator,
        )
        self.e_ema = Encoder(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            channel_multiplier=args.channel_multiplier,
        )

    def forward(self, input, mode):
        if mode == "projection":
            fake_stylecode = self.e_ema(input)

            return fake_stylecode

        elif mode == "calculate_mean_stylemap":
            truncation_mean_latent = self.g_ema(input, calculate_mean_stylemap=True)

            return truncation_mean_latent

        elif mode == "random_generation":
            z, truncation, truncation_mean_latent = input

            fake_img, _ = self.g_ema(
                z,
                truncation=truncation,
                truncation_mean_latent=truncation_mean_latent,
            )

            return fake_img

        elif mode == "reconstruction":
            fake_stylecode = self.e_ema(input)
            fake_img, _ = self.g_ema(fake_stylecode, input_is_stylecode=True)
            return fake_img

        elif mode == "stylemixing":
            w1, w2 = input
            fake_imgs = torch.Tensor().to(device)

            coarse_img, _ = self.g_ema(
                [w1, w2],
                input_is_stylecode=True,
                mix_space=f"stylemixing_coarse",
            )
            fine_img, _ = self.g_ema(
                [w1, w2], input_is_stylecode=True, mix_space=f"stylemixing_fine"
            )

            return coarse_img, fine_img

        elif mode == "w_interpolation":
            w1, w2 = input
            lambda_w = random.random()
            w = w1 * lambda_w + w2 * (1 - lambda_w)
            w = w.unsqueeze(0)
            fake_img, _ = self.g_ema(w, input_is_stylecode=True)

            return fake_img

        elif mode == "local_editing":
            w1, w2, mask = input
            w1, w2, mask = w1.unsqueeze(0), w2.unsqueeze(0), mask.unsqueeze(0)

            if dataset_name == "celeba_hq":
                mixed_image = self.g_ema(
                    [w1, w2],
                    input_is_stylecode=True,
                    mix_space="w_plus",
                    mask=mask,
                )[0]

            elif dataset_name == "afhq":
                mixed_image = self.g_ema(
                    [w1, w2], input_is_stylecode=True, mix_space="w", mask=mask
                )[0]

            recon_img_src, _ = self.g_ema(w1, input_is_stylecode=True)
            recon_img_ref, _ = self.g_ema(w2, input_is_stylecode=True)

            return mixed_image, recon_img_src, recon_img_ref

        elif mode == "transplantation":
            src_img, ref_img, coordinates = input

            src_img, ref_img = (
                src_img.unsqueeze(0),
                ref_img.unsqueeze(0),
            )
            src_w = self.e_ema(src_img)
            ref_w = self.e_ema(ref_img)
            recon_img_src, _ = self.g_ema(src_w, input_is_stylecode=True)
            recon_img_ref, _ = self.g_ema(ref_w, input_is_stylecode=True)

            for (
                (src_p_y, src_p_x),
                (ref_p_y, ref_p_x),
                height,
                width,
            ) in coordinates:
                mask_src = -torch.ones([8, 8]).to(device)
                mask_ref = -torch.ones([8, 8]).to(device)

                mask_src[src_p_y : src_p_y + height, src_p_x : src_p_x + width] = 1
                mask_ref[ref_p_y : ref_p_y + height, ref_p_x : ref_p_x + width] = 1

                mask_src, mask_ref = mask_src.unsqueeze(0), mask_ref.unsqueeze(0)
                mask_src = mask_src.unsqueeze(1).repeat(1, 64, 1, 1)
                mask_ref = mask_ref.unsqueeze(1).repeat(1, 64, 1, 1)
                src_w[mask_src == 1] = ref_w[mask_ref == 1]

            mixed_image, _ = self.g_ema(src_w, input_is_stylecode=True)

            return mixed_image, recon_img_src, recon_img_ref


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixing_type",
        choices=[
            "local_editing",
            "transplantation",
            "w_interpolation",
            "reconstruction",
            "stylemixing",
            "random_generation",
        ],
        required=True,
    )
    parser.add_argument("--ckpt", metavar="CHECKPOINT", required=True)
    parser.add_argument("--test_lmdb", type=str)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--save_image_dir", type=str, default="expr")

    # Below argument is needed for local editing.
    parser.add_argument(
        "--local_editing_part",
        type=str,
        default=None,
        choices=[
            "nose",
            "hair",
            "background",
            "eye",
            "eyebrow",
            "lip",
            "neck",
            "cloth",
            "skin",
            "ear",
        ],
    )

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    train_args = ckpt["train_args"]
    for key in vars(train_args):
        if not (key in vars(args)):
            setattr(args, key, getattr(train_args, key))
    print(args)

    dataset_name = args.dataset
    args.save_image_dir = os.path.join(
        args.save_image_dir, args.mixing_type, dataset_name
    )

    model = Model().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()

    batch = args.batch

    device = "cuda"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    if args.mixing_type == "random_generation":
        os.makedirs(args.save_image_dir, exist_ok=True)
    elif args.mixing_type in [
        "w_interpolation",
        "reconstruction",
        "transplantation",
        "stylemixing",
    ]:
        os.makedirs(args.save_image_dir, exist_ok=True)
        dataset = MultiResolutionDataset(args.test_lmdb, transform, args.size)
    elif args.mixing_type == "local_editing":

        if dataset_name == "afhq":
            args.save_image_dir = os.path.join(args.save_image_dir)
            for kind in [
                "mask",
                "source_image",
                "source_reconstruction",
                "reference_image",
                "reference_reconstruction",
                "synthesized_image",
            ]:
                os.makedirs(os.path.join(args.save_image_dir, kind), exist_ok=True)
        else:  # celeba_hq
            args.save_image_dir = os.path.join(
                args.save_image_dir,
                args.local_editing_part,
            )
            for kind in [
                "mask",
                "mask_ref",
                "mask_src",
                "source_image",
                "source_reconstruction",
                "reference_image",
                "reference_reconstruction",
                "synthesized_image",
            ]:
                os.makedirs(os.path.join(args.save_image_dir, kind), exist_ok=True)
            mask_path_base = f"data/{dataset_name}/local_editing"

        # GT celeba_hq mask images
        if dataset_name == "celeba_hq":
            assert "celeba_hq" in args.test_lmdb

            dataset = GTMaskDataset("data/celeba_hq", transform, args.size)

            parts_index = {
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

        # afhq, coarse(half-and-half) masks
        else:
            assert "afhq" in args.test_lmdb and "afhq" == dataset_name
            dataset = MultiResolutionDataset(args.test_lmdb, transform, args.size)

    if args.mixing_type in [
        "w_interpolation",
        "reconstruction",
        "stylemixing",
        "local_editing",
    ]:
        n_sample = len(dataset)
        sampler = data_sampler(dataset, shuffle=False)

        loader = data.DataLoader(
            dataset,
            batch,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # generated images should match with n sample
        if n_sample % batch == 0:
            assert len(loader) == n_sample // batch
        else:
            assert len(loader) == n_sample // batch + 1

        total_latents = torch.Tensor().to(device)
        real_imgs = torch.Tensor().to(device)

        if args.mixing_type == "local_editing":
            if dataset_name == "afhq":
                masks = (
                    -2 * torch.ones(n_sample, args.size, args.size).to(device).float()
                )

                mix_type = list(range(n_sample))
                random.shuffle(mix_type)
                horizontal_mix = mix_type[: n_sample // 2]
                vertical_mix = mix_type[n_sample // 2 :]

                masks[horizontal_mix, :, args.size // 2 :] = 2
                masks[vertical_mix, args.size // 2 :, :] = 2
            else:
                masks = torch.Tensor().to(device).long()

    with torch.no_grad():
        if args.mixing_type == "random_generation":
            truncation = 0.7
            truncation_sample = 5000
            truncation_mean_latent = torch.Tensor().to(device)
            for _ in range(truncation_sample // batch):
                z = make_noise(batch, args.latent_channel_size, device)
                partial_mean_latent = model(z, mode="calculate_mean_stylemap")
                truncation_mean_latent = torch.cat(
                    [truncation_mean_latent, partial_mean_latent], dim=0
                )
            truncation_mean_latent = truncation_mean_latent.mean(0, keepdim=True)

            # refer to stylegan official repository: https://github.com/NVlabs/stylegan/blob/master/generate_figures.py
            cx, cy, cw, ch, rows, lods = 0, 0, 1024, 1024, 3, [0, 1, 2, 2, 3, 3]

            for seed in range(0, 4):
                torch.manual_seed(seed)
                png = f"{args.save_image_dir}/random_generation_{seed}.png"
                print(png)

                total_images_len = sum(rows * 2 ** lod for lod in lods)
                total_images = torch.Tensor()

                while total_images_len > 0:
                    num = batch if total_images_len > batch else total_images_len
                    z = make_noise(num, args.latent_channel_size, device)
                    total_images_len -= batch

                    images = model(
                        (z, truncation, truncation_mean_latent),
                        mode="random_generation",
                    )

                    images = images.permute(0, 2, 3, 1)
                    images = images.cpu()
                    total_images = torch.cat([total_images, images], dim=0)

                total_images = torch.clamp(total_images, min=-1.0, max=1.0)
                total_images = (total_images + 1) / 2 * 255
                total_images = total_images.numpy().astype(np.uint8)

                canvas = Image.new(
                    "RGB",
                    (sum(cw // 2 ** lod for lod in lods), ch * rows),
                    "white",
                )
                image_iter = iter(list(total_images))
                for col, lod in enumerate(lods):
                    for row in range(rows * 2 ** lod):
                        image = Image.fromarray(next(image_iter), "RGB")
                        # image = image.crop((cx, cy, cx + cw, cy + ch))
                        image = image.resize(
                            (cw // 2 ** lod, ch // 2 ** lod), Image.ANTIALIAS
                        )
                        canvas.paste(
                            image,
                            (
                                sum(cw // 2 ** lod for lod in lods[:col]),
                                row * ch // 2 ** lod,
                            ),
                        )
                canvas.save(png)

        elif args.mixing_type == "reconstruction":
            for i, real_img in enumerate(tqdm(loader, mininterval=1)):
                real_img = real_img.to(device)
                recon_image = model(real_img, "reconstruction")

                for i_b, (img_1, img_2) in enumerate(zip(real_img, recon_image)):
                    save_images(
                        [img_1, img_2],
                        [
                            f"{args.save_image_dir}/{i*batch+i_b}_real.png",
                            f"{args.save_image_dir}/{i*batch+i_b}_recon.png",
                        ],
                    )

        elif args.mixing_type == "transplantation":

            for kind in [
                "source_image",
                "source_reconstruction",
                "reference_image",
                "reference_reconstruction",
                "synthesized_image",
            ]:
                os.makedirs(os.path.join(args.save_image_dir, kind), exist_ok=True)

            # AFHQ
            transplantation_dataset = [
                (62, 271, [((4, 2), (3, 2), 2, 4), ((0, 1), (0, 1), 3, 2)])
            ]

            for index_src, index_ref, coordinates in transplantation_dataset:
                src_img = dataset[index_src].to(device)
                ref_img = dataset[index_ref].to(device)

                mixed_image, recon_img_src, recon_img_ref = model(
                    (src_img, ref_img, coordinates), mode="transplantation"
                )

                ratio = 256 // 8

                src_img = (src_img + 1) / 2
                ref_img = (ref_img + 1) / 2

                colors = [(0, 0, 255), (0, 255, 0), (0, 255, 0)]

                for color_i, (
                    (src_p_y, src_p_x),
                    (ref_p_y, ref_p_x),
                    height,
                    width,
                ) in enumerate(coordinates):
                    for i in range(2):
                        img = src_img if i == 0 else ref_img
                        img = img.cpu()
                        img = transforms.ToPILImage()(img)
                        img = np.asarray(img)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        if i == 0:
                            img = cv2.rectangle(
                                img,
                                (src_p_x * ratio, src_p_y * ratio),
                                (
                                    (src_p_x + width) * ratio,
                                    (src_p_y + height) * ratio,
                                ),
                                colors[color_i],
                                2,
                            )
                        else:
                            img = cv2.rectangle(
                                img,
                                (ref_p_x * ratio, ref_p_y * ratio),
                                (
                                    (ref_p_x + width) * ratio,
                                    (ref_p_y + height) * ratio,
                                ),
                                colors[color_i],
                                2,
                            )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = transforms.ToTensor()(img)

                        if i == 0:
                            src_img = img
                        else:
                            ref_img = img

                save_images(
                    [mixed_image[0], recon_img_src[0], recon_img_ref[0]],
                    [
                        f"{args.save_image_dir}/synthesized_image/{index_src}_{index_ref}.png",
                        f"{args.save_image_dir}/source_reconstruction/{index_src}_{index_ref}.png",
                        f"{args.save_image_dir}/reference_reconstruction/{index_src}_{index_ref}.png",
                    ],
                )

                save_images(
                    [src_img, ref_img],
                    [
                        f"{args.save_image_dir}/source_image/{index_src}_{index_ref}.png",
                        f"{args.save_image_dir}/reference_image/{index_src}_{index_ref}.png",
                    ],
                    range=(0, 1),
                )

        else:
            for i, real_img in enumerate(tqdm(loader, mininterval=1)):
                if (args.mixing_type == "local_editing") and (
                    dataset_name == "celeba_hq"
                ):
                    real_img, mask = real_img
                    mask = mask.to(device)
                    masks = torch.cat([masks, mask], dim=0)
                real_img = real_img.to(device)

                latents = model(real_img, "projection")

                total_latents = torch.cat([total_latents, latents], dim=0)
                real_imgs = torch.cat([real_imgs, real_img], dim=0)

            if args.mixing_type == "stylemixing":
                with open(
                    "data/celeba_hq/local_editing/celeba_hq_test_GT_sorted_pair.pkl",
                    "rb",
                ) as f:
                    sorted_similarity = pickle.load(f)

                indices1 = []
                indices2 = []
                reference = [356, 358, 363, 400, 483]
                original = [7, 124, 162, 135, 136, 173]
                for i in original:
                    indices1.append(i)
                    indices2.append(reference)

            elif args.mixing_type == "w_interpolation":
                indices = list(range(len(total_latents)))

                if dataset_name == "celeba_hq":
                    n_sample = 29000
                elif dataset_name == "afhq":
                    n_sample = 15130
                elif dataset_name == "ffhq":
                    n_sample = 69000

                indices1 = random.choices(indices, k=n_sample)
                indices2 = random.choices(indices, k=n_sample)

            elif args.mixing_type == "local_editing":
                if dataset_name == "afhq":
                    # change it later
                    indices = list(range(len(total_latents)))
                    random.shuffle(indices)
                    indices1 = indices[: len(total_latents) // 2]
                    indices2 = indices[len(total_latents) // 2 :]

                else:
                    with open(
                        f"{mask_path_base}/celeba_hq_test_GT_sorted_pair.pkl",
                        "rb",
                    ) as f:
                        sorted_similarity = pickle.load(f)

                    indices1 = []
                    indices2 = []
                    for (i1, i2), _ in sorted_similarity[args.local_editing_part]:
                        indices1.append(i1)
                        indices2.append(i2)

            for loop_i, (index1, index2) in tqdm(
                enumerate(zip(indices1, indices2)), total=n_sample
            ):
                if args.mixing_type == "w_interpolation":
                    imgs = model(
                        (total_latents[index1], total_latents[index2]),
                        "w_interpolation",
                    )
                    assert len(imgs) == 1
                    save_image(
                        imgs[0],
                        f"{args.save_image_dir}/{loop_i}.png",
                    )
                elif args.mixing_type == "stylemixing":
                    n_rows = len(index2)
                    coarse_img, fine_img = model(
                        (
                            torch.stack([total_latents[index1] for _ in range(n_rows)]),
                            torch.stack([total_latents[i2] for i2 in index2]),
                        ),
                        "stylemixing",
                    )

                    save_images(
                        [coarse_img, fine_img],
                        [
                            f"{args.save_image_dir}/{index1}_coarse.png",
                            f"{args.save_image_dir}/{index1}_fine.png",
                        ],
                    )

                elif args.mixing_type == "local_editing":
                    src_img = real_imgs[index1]
                    ref_img = real_imgs[index2]

                    if dataset_name == "celeba_hq":
                        mask1_logit = masks[index1]
                        mask2_logit = masks[index2]

                        mask1 = -torch.ones(mask1_logit.shape).to(
                            device
                        )  # initialize with -1
                        mask2 = -torch.ones(mask2_logit.shape).to(
                            device
                        )  # initialize with -1

                        for label_i in parts_index[args.local_editing_part]:
                            mask1[(mask1_logit == label_i) == True] = 1
                            mask2[(mask2_logit == label_i) == True] = 1

                        mask = mask1 + mask2
                        mask = mask.float()
                    elif dataset_name == "afhq":
                        mask = masks[index1]

                    mixed_image, recon_img_src, recon_img_ref = model(
                        (total_latents[index1], total_latents[index2], mask),
                        "local_editing",
                    )

                    save_images(
                        [
                            mixed_image[0],
                            recon_img_src[0],
                            src_img,
                            ref_img,
                            recon_img_ref[0],
                        ],
                        [
                            f"{args.save_image_dir}/synthesized_image/{index1}.png",
                            f"{args.save_image_dir}/source_reconstruction/{index1}.png",
                            f"{args.save_image_dir}/source_image/{index1}.png",
                            f"{args.save_image_dir}/reference_image/{index1}.png",
                            f"{args.save_image_dir}/reference_reconstruction/{index1}.png",
                        ],
                    )

                    mask[mask < -1] = -1
                    mask[mask > -1] = 1

                    save_image(
                        mask,
                        f"{args.save_image_dir}/mask/{index1}.png",
                    )

                    if dataset_name == "celeba_hq":
                        save_images(
                            [mask1, mask2],
                            [
                                f"{args.save_image_dir}/mask_src/{index1}.png",
                                f"{args.save_image_dir}/mask_ref/{index1}.png",
                            ],
                        )