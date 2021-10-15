"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

CMD=$1
DATASET=$2

if [ $CMD == "prepare-fid-calculation" ]; then 
    # download pretrained network and stats(mean, var) of each dataset to calculate FID
    URL="https://docs.google.com/uc?export=download&id=1pCr4lNCON7IZcNVdskIDXhFJ3jYuge1w"
    NETWORK_FOLDER="./metrics"
    NETWORK_FILE=$NETWORK_FOLDER/pt_inception-2015-12-05-6726825d.pth
    wget --no-check-certificate $URL -O $NETWORK_FILE

    # download precalculated statistics in several datasets
    URL="https://docs.google.com/uc?export=download&id=1sJ7AYaY3JTVFqI6Dodnzx81KydadRpnj"
    mkdir -p "./metrics/fid_stats"
    ZIP_FILE="./metrics/fid_stats.zip"
    wget --no-check-certificate -r $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d "./metrics/fid_stats"
    rm $ZIP_FILE
    
elif [ $CMD == "create-lmdb-dataset" ]; then 
    if [ $DATASET == "celeba_hq" ]; then
        URL="https://docs.google.com/uc?export=download&id=1R72NB79CX0MpnmWSli2SMu-Wp-M0xI-o"
        DATASET_FOLDER="./data/celeba_hq"
        ZIP_FILE=$DATASET_FOLDER/celeba_hq_raw.zip
    elif  [ $DATASET == "afhq" ]; then
        URL="https://docs.google.com/uc?export=download&id=1Pf4f6Y27lQX9y9vjeSQnoOQntw_ln7il"
        DATASET_FOLDER="./data/afhq"
        ZIP_FILE=$DATASET_FOLDER/afhq_raw.zip
    else
        echo "Unknown DATASET"
        exit 1
    fi
    mkdir -p $DATASET_FOLDER
    wget --no-check-certificate -r $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d $DATASET_FOLDER
    rm $ZIP_FILE

    # raw images to LMDB format
    TARGET_SIZE=256,1024
    for DATASET_TYPE in "train" "test" "val"; do
        python preprocessor/prepare_data.py --out $DATASET_FOLDER/LMDB_$DATASET_TYPE --size $TARGET_SIZE $DATASET_FOLDER/raw_images/$DATASET_TYPE
    done
    
    # for local editing
    FOLDERNAME=$DATASET_FOLDER/local_editing
    mkdir -p $FOLDERNAME

    if [ $DATASET == "celeba_hq" ]; then
        URL="https://docs.google.com/uc?export=download&id=1_4Cxd7uH8Zqlutu5zUNJfVpgljqh7Olf"
        wget -r --no-check-certificate $URL -O $FOLDERNAME/GT_labels.zip
        unzip $FOLDERNAME/GT_labels.zip -d $FOLDERNAME
        rm $FOLDERNAME/GT_labels.zip
        URL="https://docs.google.com/uc?export=download&id=1dy-3UxETpI58xroeAGXqidSxRCgV71SV"
        wget -r --no-check-certificate $URL -O $FOLDERNAME/LMDB_test_mask.zip
        unzip $FOLDERNAME/LMDB_test_mask.zip -d $FOLDERNAME
        rm $FOLDERNAME/LMDB_test_mask.zip
        URL="https://docs.google.com/uc?export=download&id=1rCxK0ybho9Xqexvec0g0khPPLL_cRZfx"
        wget --no-check-certificate $URL -O $FOLDERNAME/celeba_hq_test_GT_sorted_pair.pkl
        URL="https://docs.google.com/uc?export=download&id=1g4tatzpPsycq2h4B2NjejgiuHUGIvuB0"
        wget --no-check-certificate $URL -O $FOLDERNAME/CelebA-HQ-to-CelebA-mapping.txt
    fi 

elif  [ $CMD == "download-pretrained-network-256" ]; then
    # 20M-image-trained models
    if [ $DATASET == "celeba_hq" ]; then
        URL="https://docs.google.com/uc?export=download&id=1Up6qELYFF1cV0HREnHpykKN2Ordr1xpp"
    elif  [ $DATASET == "afhq" ]; then
        URL="https://docs.google.com/uc?export=download&id=1gKSxyBWUc53OaRwFZ6w2CuNkiLf3_X9C"
    elif  [ $DATASET == "lsun_car" ]; then
        URL="https://docs.google.com/uc?export=download&id=1P77_21yBcgF5AMs8hMBT8m0eEhR5j8_Q"
    elif  [ $DATASET == "lsun_church" ]; then
        URL="https://docs.google.com/uc?export=download&id=1sxdDn2dK1Ilqv9KSrXqABSUywV25pbin"

    # 5M-image-trained models used in our paper for comparison with other baselines and for ablation studies.
    elif [ $DATASET == "celeba_hq_5M" ]; then
        URL="https://docs.google.com/uc?export=download&id=1-t9WkasJzn4-pljZI5619SMyvDXB_9iv"
    elif  [ $DATASET == "afhq_5M" ]; then
        URL="https://docs.google.com/uc?export=download&id=1on4L_2WAl8PpH4iU1wrRCftbE_j_CrTI"
    else
        echo "Unknown DATASET"
        exit 1
    fi
    
    NETWORK_FOLDER="./expr/checkpoints"
    NETWORK_FILE=$NETWORK_FOLDER/${DATASET}_256_8x8.pt
    mkdir -p $NETWORK_FOLDER
    wget -r --no-check-certificate $URL -O $NETWORK_FILE

elif  [ $CMD == "download-pretrained-network-1024" ]; then
    NETWORK_FOLDER="./expr/checkpoints"
    mkdir -p $NETWORK_FOLDER
    if [ $DATASET == "ffhq_16x16" ]; then
        URL="https://docs.google.com/uc?export=download&id=14wPqqpWIe34hh2LHoSsOhXidrm4bS3Sg"
        NETWORK_FILE=$NETWORK_FOLDER/ffhq_1024_16x16.pt
    elif  [ $DATASET == "ffhq_32x32" ]; then
        URL="https://docs.google.com/uc?export=download&id=1UqBHEICkL1Ml2m56eG3_u9_KvddeJlDK"
        NETWORK_FILE=$NETWORK_FOLDER/ffhq_1024_32x32.pt     
    else
        echo "Unknown DATASET"
        exit 1
    fi
    wget -r --no-check-certificate $URL -O $NETWORK_FILE

else
    echo "Unknown CMD"
    exit 1
fi
