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
import os
import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm
from training.model import Generator
from metrics.calc_inception import load_patched_inception_v3
from torchvision import utils, transforms
from training.dataset import MultiResolutionDataset, DataSetFromDir
from torch.utils import data
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset


class DPModel(nn.Module):
    def __init__(self, device, model_args):
        super(DPModel, self).__init__()
        self.g_ema = Generator(
            model_args.size,
            model_args.mapping_layer_num,
            model_args.latent_channel_size,
            model_args.latent_spatial_size,
            lr_mul=model_args.lr_mul,
            channel_multiplier=model_args.channel_multiplier,
            normalize_mode=model_args.normalize_mode,
            small_generator=model_args.small_generator,
        )

    def forward(self, real_img):
        z = real_img
        fake_img, _ = self.g_ema(z)

        return fake_img


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def make_noise(batch, latent_channel_size, device):
    return torch.randn(batch, latent_channel_size, device=device)


@torch.no_grad()
def extract_feature_from_samples(generator, inception, batch_size, n_sample, device):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    if resid > 0:
        batch_sizes = [batch_size] * n_batch + [resid]
    else:
        batch_sizes = [batch_size] * n_batch
    features = []

    for batch in tqdm(batch_sizes):
        latent = make_noise(batch, train_args.latent_channel_size, device)
        imgs = generator(latent)
        imgs = (imgs + 1) / 2  # -1 ~ 1 to 0~1
        imgs = torch.clamp(imgs, 0, 1, out=None)
        imgs = F.interpolate(imgs, size=(height, width), mode="bilinear")
        transformed = []

        for img in imgs:
            transformed.append(transforms.Normalize(mean=mean, std=std)(img))

        transformed = torch.stack(transformed, dim=0)

        assert transformed.shape == imgs.shape
        feat = inception(transformed)[0].view(imgs.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


def extract_feature_from_generated_samples(
    inception, batch_size, n_sample, device, transform
):

    features = []

    try:  # from LMDB
        dataset = MultiResolutionDataset(
            args.generated_image_path, transform, args.size
        )
    except:  # from raw images
        dataset = DataSetFromDir(args.generated_image_path, transform)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data_sampler(dataset, shuffle=True),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # generated images should match with n sample
    print(len(loader), n_sample, batch_size)

    if n_sample % batch_size == 0:
        assert len(loader) == n_sample // batch_size
    else:
        assert len(loader) == n_sample // batch_size + 1

    for i, real_img in enumerate(tqdm(loader)):
        real_img = real_img.to(device)

        if args.batch * (i + 1) > n_sample:
            real_img = real_img[: n_sample - args.batch * i]

        feat = inception(real_img)[0].view(real_img.shape[0], -1)
        features.append(feat.to("cpu"))

        if args.batch * (i + 1) > n_sample:
            break

    features = torch.cat(features, 0)
    print(len(features))
    assert len(features) == n_sample

    return features


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--comparative_fid_pkl", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--generated_image_path", type=str)
    parser.add_argument("--ckpt", metavar="CHECKPOINT")
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()
    assert ((args.generated_image_path is None) and (args.ckpt is not None)) or (
        (args.generated_image_path is not None) and (args.ckpt is None)
    )

    if args.dataset == "celeba_hq":
        n_sample = 29000
    elif args.dataset == "afhq":
        n_sample = 15130
    elif args.dataset in ["lsun/car", "lsun/church_outdoor"]:
        n_sample = 50000
    elif args.dataset == "ffhq":
        n_sample = 69000

    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()
    height, width = 299, 299
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if args.ckpt:
        assert args.comparative_fid_pkl is not None

        ckpt = torch.load(args.ckpt)
        train_args = ckpt["train_args"]
        assert args.size == train_args.size
        model = DPModel(device, train_args).to(device)
        model.g_ema.load_state_dict(ckpt["g_ema"])
        model = nn.DataParallel(model)
        model.eval()

        features = extract_feature_from_samples(
            model, inception, args.batch, n_sample, device
        ).numpy()

    else:
        transform = transforms.Compose(
            [
                transforms.Resize([args.size, args.size]),
                transforms.Resize([height, width]),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        features = extract_feature_from_generated_samples(
            inception, args.batch, n_sample, device, transform
        ).numpy()

    print(f"extracted {features.shape[0]} features")

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    if (args.generated_image_path is not None) and (args.comparative_fid_pkl is None):
        with open(
            f"metrics/fid_stats/{args.dataset}_stats_{args.size}_{n_sample}.pkl",
            "wb",
        ) as handle:
            pickle.dump({"mean": sample_mean, "cov": sample_cov}, handle)
    else:
        with open(args.comparative_fid_pkl, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]

        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

        print("fid:", fid)
