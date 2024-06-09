# Requirements

Python 3.6.5, Pytorch 1.9.0, Numpy 1.16.3, argparse and configparser

# Model Training

```bash
# METR
python train.py --dataset METR --adjdata data/adj_mx.pkl

# PeMS
python train.py --dataset PeMS --adjdata data/adj_mx_bay.pkl

#BJ
python train_BJ.py 
```

# Data file

download the data file from the following link and add it to the RGDAN directory

https://drive.google.com/drive/folders/1rO3wWrx0Uqjij8uKkaHjTqZarGmO79-i?usp=sharing
