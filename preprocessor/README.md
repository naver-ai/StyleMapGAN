## Dataset

Transform raw images(ex. jpg, png) to LMDB format. Refer to `download.sh`.
```
python prepare_data.py [raw images path] --out [destination path] --size [TARGET_SIZE] 

```

## Local editing

We use an overall mask of original and reference mask, so we need a pair of images which has a similar target mask with each other. `download.sh create-lmdb-dataset celeba_hq` already offers precalculated pairs of images for local editing. But you can pair your own images based on the similarity of target semantic(e.g., nose, hair) mask, please modify `pair_masks.py` for your purposes.
```
python pair_masks.py
```