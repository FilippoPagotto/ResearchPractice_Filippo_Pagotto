# [Neural Networks] RGDAN: A random graph diffusion attention network for traffic prediction  

This is a PyTorch implementation of Decomposition Dynamic Graph Conolutional Recurrent Network for Traffic Forecasting, as described in our paper: Jin Fan, [Weng, Wenchao](https://github.com/wengwenchao123/RGDAN/), Hao Tian, Huifeng Wu , Fu Zhu, Jia Wu **[RGDAN: A random graph diffusion attention network for traffic prediction](https://doi.org/10.1016/j.neunet.2023.106093)**,Neural Networks 2024.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rgdan-a-random-graph-diffusion-attention/traffic-prediction-on-metr-la)](https://paperswithcode.com/sota/traffic-prediction-on-metr-la?p=rgdan-a-random-graph-diffusion-attention)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rgdan-a-random-graph-diffusion-attention/traffic-prediction-on-pems-bay)](https://paperswithcode.com/sota/traffic-prediction-on-pems-bay?p=rgdan-a-random-graph-diffusion-attention)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rgdan-a-random-graph-diffusion-attention/traffic-prediction-on-ne-bj)](https://paperswithcode.com/sota/traffic-prediction-on-ne-bj?p=rgdan-a-random-graph-diffusion-attention)
## Note
The original code for this paper was lost due to server damage a year ago, and there was a lack of awareness to save relevant data at that time. The current code has been reconstructed based on memory to provide a version for research reference. While it achieves good results, it may not match the performance reported in the paper due to unknown reasons. We appreciate your understanding.

# Data Preparation

The relevant datasets have been placed in the "data" folder. To run the program, simply unzip the "PeMS.zip" and "METR.zip" files.

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


## Cite

If you find the paper useful, please cite as following:

```
@article{fan2024rgdan,
  title={RGDAN: A random graph diffusion attention network for traffic prediction},
  author={Fan, Jin and Weng, Wenchao and Tian, Hao and Wu, Huifeng and Zhu, Fu and Wu, Jia},
  journal={Neural networks},
  pages={106093},
  year={2024},
  publisher={Elsevier}
}
```

## More Related Works

- [[Pattern Recognition] A Decomposition Dynamic Graph Convolutional Recurrent Network for Traffic Forecasting](https://www.sciencedirect.com/science/article/pii/S0031320323003710)



# The code defines a variety of components for a Spatial-Temporal Model using PyTorch, including attention mechanisms, fusion models, and loss functions. Each component serves a specific purpose:

# 1. **Graph Convolutional Network (GCN)**:
#     - Defined by the `gcn` class, it takes inputs X, STE (Spatial-Temporal Embedding), and A (Adjacency Matrix) and performs graph convolution operations.
#     - `forward` method concatenates X and STE, applies linear transformation, and then performs matrix multiplication with the adjacency matrix A.

# 2. **Random Graph Attention Network (GAT)**:
#     - Defined by the `randomGAT` class, it utilizes attention mechanisms to process spatial-temporal data.
#     - `forward` method concatenates X and STE, applies linear transformation, calculates attention weights using learned node embeddings, and performs attention-weighted aggregation.

# 3. **Spatial-Temporal Embedding Model**:
#     - Defined by the `STEmbModel` class, it embeds spatial and temporal features.
#     - `forward` method processes spatial and temporal features through linear layers and activation functions.

# 4. **Spatial Attention Model**:
#     - Defined by the `SpatialAttentionModel` class, it employs attention mechanisms to capture spatial dependencies in the data.
#     - `forward` method calculates attention weights based on query, key, and value embeddings, and performs attention-weighted aggregation.

# 5. **Temporal Attention Model**:
#     - Defined by the `TemporalAttentionModel` class, it captures temporal dependencies in the data using attention mechanisms.
#     - `forward` method calculates attention weights between query and key embeddings and performs attention-weighted aggregation.

# 6. **Gated Fusion Model**:
#     - Defined by the `GatedFusionModel` class, it fuses spatial and temporal information using gated mechanisms.
#     - `forward` method computes gates based on spatial and temporal embeddings, combines them, and applies further transformations.

# 7. **Spatial-Temporal Attention Model**:
#     - Defined by the `STAttModel` class, it combines spatial and temporal attention mechanisms for comprehensive modeling.
#     - `forward` method incorporates multiple instances of GATs and temporal attention, and fuses their outputs using a gated fusion model.

# 8. **Transform Attention Model**:
#     - Defined by the `TransformAttentionModel` class, it employs attention mechanisms to transform embeddings.
#     - `forward` method calculates attention weights between query and key embeddings and performs attention-weighted aggregation.

# 9. **Spatial-Temporal Model**:
#     - Defined by the `SpatialTemporalModel` class, it integrates spatial and temporal information using a combination of LSTM and RGDAN.
#     - `forward` method processes input sequences through LSTM, then through the RGDAN model for spatial-temporal fusion.

# 10. **Residual Gated Dynamic Attention Network (RGDAN)**:
#     - Defined by the `RGDAN` class, it combines various components such as spatial-temporal embedding, attention blocks, and fusion mechanisms for effective modeling.
#     - `forward` method orchestrates the flow of data through the network, applying spatial-temporal attention and fusion.

# 11. **Loss Functions**:
#     - `mae_loss`, `mse_loss`, and `mape_loss` define Mean Absolute Error, Mean Squared Error, and Mean Absolute Percentage Error loss functions respectively. They account for zero-padded elements in the input data to ensure accurate loss calculation.

# These components collectively enable the construction of a sophisticated Spatial-Temporal Model capable of capturing complex spatio-temporal patterns in the data.