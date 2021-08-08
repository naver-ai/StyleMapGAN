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
    URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/ETSCIvNVUKBDk_mF4Ed2rj8BFajfPC5apv19PrCj6LUr6g?e=RCCHjq&download=1"
    NETWORK_FOLDER="./metrics"
    NETWORK_FILE=$NETWORK_FOLDER/pt_inception-2015-12-05-6726825d.pth
    wget -N $URL -O $NETWORK_FILE

    # download precalculated statistics in several datasets
    URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EWJvmMFTIhdHn1ZIInuNUe4BX_d1ZD4EUfrBXQSgGH7d0w?e=5LUOBi&download=1"
    mkdir -p "./metrics/fid_stats"
    ZIP_FILE="./metrics/fid_stats.zip"
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d "./metrics/fid_stats"
    rm $ZIP_FILE
    
elif [ $CMD == "create-lmdb-dataset" ]; then 
    if [ $DATASET == "celeba_hq" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EQANtAapq_5Pu9nElWECb_IBGdGhvODhRJRnKSEPlzHaZw?e=Pr3Csz&download=1"
        DATASET_FOLDER="./data/celeba_hq"
        ZIP_FILE=$DATASET_FOLDER/celeba_hq_raw.zip
    elif  [ $DATASET == "afhq" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EbPgjpgkxDdBsAqS5rhSqr4BvbWM-MuWKuTSPl1EYPzCcw?e=eCm3tT&download=1"
        DATASET_FOLDER="./data/afhq"
        ZIP_FILE=$DATASET_FOLDER/afhq_raw.zip
    else
        echo "Unknown DATASET"
        exit 1
    fi
    mkdir -p $DATASET_FOLDER
    wget -N $URL -O $ZIP_FILE
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
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EZ5ZFSNaTLpGpHgmJU88Rm4B3ysk28KB2x-Ms8ax_bHlMg?e=tuzwSF&download=1"
        wget -r -N $URL -O $FOLDERNAME/GT_labels.zip
        unzip $FOLDERNAME/GT_labels.zip -d $FOLDERNAME
        rm $FOLDERNAME/GT_labels.zip
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EQM4ljzZo45JmOnlDhICUWwB_5c4RyxEAMz06MgejMYfRQ?e=kxvfDh&download=1"
        wget -N $URL -O $FOLDERNAME/LMDB_test_mask.zip
        wget -r -N $URL -O $FOLDERNAME/LMDB_test_mask.zip
        unzip $FOLDERNAME/LMDB_test_mask.zip -d $FOLDERNAME
        rm $FOLDERNAME/LMDB_test_mask.zip
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EYZLFOrVxGxKuTQj5aIUPpQBcTiyGCkldDP4R1Vwy1dF4Q?e=J1C7Mj&download=1"
        wget -N $URL -O $FOLDERNAME/celeba_hq_test_GT_sorted_pair.pkl
        URL="https://mysnu-my.sharepoint.com/:t:/g/personal/gustnxodjs_seoul_ac_kr/EYl48QPNNPhHnV9kzQ120asBVg0M8E5dedpIEYJAyO9lCg?e=zI6N7r&download=1"
        wget -N $URL -O $FOLDERNAME/CelebA-HQ-to-CelebA-mapping.txt
    fi 

elif  [ $CMD == "download-pretrained-network-256" ]; then
    # 20M-image-trained models
    if [ $DATASET == "celeba_hq" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EeNLqY2QJhdIiaLn2eoQWB0B0SsuDpkBlKI-Dd2HxiQ2dg?e=7KXIcE&download=1"
    elif  [ $DATASET == "afhq" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EWs_zF9FhNpLq9_qTvMWr98BA0AQEoItdrd2ZO-gD3xuCA?e=1TfZSU&download=1"
    elif  [ $DATASET == "lsun_car" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EYwE8gwJtSdCmzhaenyvl0wBdIevFSUj8XMbo0YPc-J_qg?e=luroy9&download=1"
    elif  [ $DATASET == "lsun_church" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/ETflqhQF2h9Dp7yD2pz2-64BFLq1bs5mJK42nRJqqviWig?e=0zNFe8&download=1"

    # 5M-image-trained models used in our paper for comparison with other baselines and for ablation studies.
    elif [ $DATASET == "celeba_hq_5M" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/ERjs1Hj0vvJPiQ6d-Yew_qsBi9c-PT-xZygSeO-Nvixeug?e=SzeeJb&download=1"
    elif  [ $DATASET == "afhq_5M" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EebeiMTHbK5Irw1allTdp0oBgpu_m8AXOJPqNYe1m5nJPw?e=xu1TlH&download=1"
    else
        echo "Unknown DATASET"
        exit 1
    fi
    
    NETWORK_FOLDER="./expr/checkpoints"
    NETWORK_FILE=$NETWORK_FOLDER/${DATASET}_256_8x8.pt
    mkdir -p $NETWORK_FOLDER
    wget -N $URL -O $NETWORK_FILE

elif  [ $CMD == "download-pretrained-network-1024" ]; then
    NETWORK_FOLDER="./expr/checkpoints"
    mkdir -p $NETWORK_FOLDER
    if [ $DATASET == "ffhq_16x16" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/Ec1770ZJlvBEum-IBA93IKIBUrp9nnZ_BX8BnTvap_8MQA?e=6yjzeM&download=1"
        NETWORK_FILE=$NETWORK_FOLDER/ffhq_1024_16x16.pt
    elif  [ $DATASET == "ffhq_32x32" ]; then
        URL="https://mysnu-my.sharepoint.com/:u:/g/personal/gustnxodjs_seoul_ac_kr/EbSmtpWIsZdPhvQZeVFPgNIBnfvEl6OOcbnoQqhFYt6SpA?e=wztFAt&download=1"
        NETWORK_FILE=$NETWORK_FOLDER/ffhq_1024_32x32.pt     
    else
        echo "Unknown DATASET"
        exit 1
    fi
    wget -N $URL -O $NETWORK_FILE

else
    echo "Unknown CMD"
    exit 1
fi
