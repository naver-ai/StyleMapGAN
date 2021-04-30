#!/bin/bash

conda install -y pytorch=1.4.0 torchvision=0.5.0 -c pytorch
conda install -y numpy=1.18.1 scikit-image=0.16.2 tqdm
conda install -y -c anaconda ipython=7.13.0
pip install lmdb==0.98 opencv-python==4.2.0.34 munch==2.5.0
pip install -U scikit-image==0.15.0 scipy==1.2.1 matplotlib scikit-learn
pip install flask==1.0.2 pillow==7.0.0