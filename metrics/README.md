## Metrics

* Reconstruction: LPIPS, MSE
* W interpolation: FID<sub>lerp</sub>
* Generation: FID from Gaussian distribution
* Local editing: MSE<sub>src</sub>, MSE<sub>ref</sub> / Detectability (Refer to [CNNDetection](https://github.com/PeterWang512/CNNDetection))


All command lines should be run in `StyleMapGAN/`
```bash
bash download.sh prepare-fid-calculation
```

<b>Reconstruction</b>
```bash
# First, reconstruct images
# python generate.py --ckpt expr/checkpoints/celeba_hq_256_8x8.pt --mixing_type reconstruction --test_lmdb data/celeba_hq/LMDB_test
python -m metrics.reconstruction --image_folder_path expr/reconstruction/celeba_hq
```

<b>W interpolation</b>
```bash
# First, interpolate images
# python generate.py --ckpt expr/checkpoints/celeba_hq_256_8x8.pt --mixing_type w_interpolation --test_lmdb data/celeba_hq/LMDB_test

# Second, precalculate mean and variance of dataset: fid/celeba_hq_stats_256_29000.pkl 
# But, we already provided them.
# python -m metrics.fid --size 256 --dataset celeba_hq --generated_image_path data/celeba_hq/LMDB_train

# CelebA-HQ
python -m metrics.fid --comparative_fid_pkl metrics/fid_stats/celeba_hq_stats_256_29000.pkl --dataset celeba_hq --generated_image_path expr/w_interpolation/celeba_hq 

# AFHQ
python -m metrics.fid --comparative_fid_pkl metrics/fid_stats/afhq_stats_256_15130.pkl --dataset afhq --generated_image_path expr/w_interpolation/afhq 
```

<b>Generation</b>
```bash
python -m metrics.fid --ckpt expr/checkpoints/celeba_hq_256_8x8.pt --comparative_fid_pkl metrics/fid_stats/celeba_hq_stats_256_29000.pkl --dataset celeba_hq
```

<b>Local editing</b>

For MSE<sub>src</sub>, MSE<sub>ref</sub>
```bash
# First, generate local edited image
# for part in nose hair background eye eyebrow lip neck cloth skin ear; do
#     python generate.py --ckpt expr/checkpoints/celeba_hq_256_8x8.pt --mixing_type local_editing --test_lmdb data/celeba_hq/LMDB_test --local_editing_part $part
# done
python -m metrics.local_editing --data_dir expr/local_editing/celeba_hq
```