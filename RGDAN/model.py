import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# Define a Graph Convolutional Network (GCN) module
class gcn(torch.nn.Module):
    def __init__(self, k, d):
        super(gcn, self).__init__()
        D = k * d
        # Fully connected layer for GCN
        self.fc = torch.nn.Linear(2 * D, D)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.1)

        # Xavier initialization for the linear layer
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, X, STE, A):
        # Concatenate input features X and spatial-temporal embedding (STE)
        X = torch.cat((X, STE), dim=-1)
        # Pass through fully connected layer and apply GELU activation function
        H = F.gelu(self.fc(X))
        # Perform graph convolution using matrix multiplication with adjacency matrix A
        H = torch.einsum('ncvl,vw->ncwl', (H, A))
        # Apply dropout
        return self.dropout(H.contiguous())

# Define a random Graph Attention Network (GAT) module
class randomGAT(torch.nn.Module):
    def __init__(self, k, d, adj, device):
        super(randomGAT, self).__init__()
        D = k * d
        self.d = d
        self.K = k
        num_nodes = adj.shape[0]
        self.device = device
        # Fully connected layer for GAT
        self.fc = torch.nn.Linear(2 * D, D)
        self.adj = adj
        # Node embedding parameters
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

        # Xavier initialization for the linear layer
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, X, STE):
        X = torch.cat((X, STE), dim=-1)
        H = F.gelu(self.fc(X))
        H = torch.cat(torch.split(H, self.d, dim=-1), dim=0)
        adp = torch.mm(self.nodevec1, self.nodevec2)
        zero_vec = torch.tensor(-9e15).to(self.device)
        # Apply mask to the attention mechanism
        adp = torch.where(self.adj > 0, adp, zero_vec)
        adj = F.softmax(adp, dim=-1)
        H = torch.einsum('vw,ncwl->ncvl', (adj, H))
        H = torch.cat(torch.split(H, H.shape[0] // self.K, dim=0), dim=-1)
        return F.gelu(H.contiguous())

# Define a Spatial-Temporal Embedding Model
class STEmbModel(torch.nn.Module):
    def __init__(self, SEDims, TEDims, OutDims, device):
        super(STEmbModel, self).__init__()
        self.TEDims = TEDims
        # Fully connected layers for spatial-temporal embedding
        self.fc3 = torch.nn.Linear(SEDims, OutDims)
        self.fc4 = torch.nn.Linear(OutDims, OutDims)
        self.fc5 = torch.nn.Linear(TEDims, OutDims)
        self.fc6 = torch.nn.Linear(OutDims, OutDims)
        self.device = device

        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, SE, TE):
        # Process spatial features
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.fc4(F.gelu(self.fc3(SE)))
        # Process temporal features
        dayofweek = F.one_hot(TE[..., 0], num_classes=7)
        timeofday = F.one_hot(TE[..., 1], num_classes=self.TEDims - 7)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(2).type(torch.FloatTensor).to(self.device)
        TE = self.fc6(F.gelu(self.fc5(TE)))
        # Combine spatial and temporal features
        sum_tensor = torch.add(SE, TE)
        return sum_tensor

# Define a Spatial Attention Model
class SpatialAttentionModel(torch.nn.Module):
    def __init__(self, K, d, adj, dropout=0.3, mask=True):
        super(SpatialAttentionModel, self).__init__()
        D = K * d
        self.fc7 = torch.nn.Linear(2 * D, D)
        self.fc8 = torch.nn.Linear(2 * D, D)
        self.fc9 = torch.nn.Linear(2 * D, D)
        self.fc10 = torch.nn.Linear(D, D)
        self.fc11 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.adj = adj
        self.mask = mask
        self.dropout = dropout
        self.softmax = torch.nn.Softmax(dim=-1)

        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.fc7.weight)
        nn.init.xavier_uniform_(self.fc8.weight)
        nn.init.xavier_uniform_(self.fc9.weight)
        nn.init.xavier_uniform_(self.fc10.weight)
        nn.init.xavier_uniform_(self.fc11.weight)

    def forward(self, X, STE):
        X = torch.cat((X, STE), dim=-1)
        query = F.gelu(self.fc7(X))
        key = F.gelu(self.fc8(X))
        value = F.gelu(self.fc9(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        attention = torch.matmul(query, torch.transpose(key, 2, 3))
        attention /= (self.d ** 0.5)
        if self.mask:
            zero_vec = -9e15 * torch.ones_like(attention)
            attention = torch.where(self.adj > 0, attention, zero_vec)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, X.shape[0] // self.K, dim=0), dim=-1)
        X = self.fc11(F.gelu(self.fc10(X)))
        return X

# Define a Temporal Attention Model
class TemporalAttentionModel(torch.nn.Module):
    def __init__(self, K, d, device):
        super(TemporalAttentionModel, self).__init__()
        D = K * d
        self.fc12 = torch.nn.Linear(2 * D, D)
        self.fc13 = torch.nn.Linear(2 * D, D)
        self.fc14 = torch.nn.Linear(2 * D, D)
        self.fc15 = torch.nn.Linear(D, D)
        self.fc16 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)

        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.fc12.weight)
        nn.init.xavier_uniform_(self.fc13.weight)
        nn.init.xavier_uniform_(self.fc14.weight)
        nn.init.xavier_uniform_(self.fc15.weight)
        nn.init.xavier_uniform_(self.fc16.weight)

    def forward(self, X, STE, Mask=True):
        X = torch.cat((X, STE), dim=-1)
        query = F.gelu(self.fc12(X))
        key = F.gelu(self.fc13(X))
        value = F.gelu(self.fc14(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        if Mask == True:
            batch_size = X.shape[0]
            num_steps = X.shape[1]
            num_vertexs = X.shape[2]
            mask = torch.ones(num_steps, num_steps).to(self.device)
            mask = torch.tril(mask)
            zero_vec = torch.tensor(-9e15).to(self.device)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, zero_vec)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, X.shape[0] // self.K, dim=0), dim=-1)
        X = self.dropout(self.fc16(F.gelu(self.fc15(X))))
        return X

# Define a Gated Fusion Model
class GatedFusionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(GatedFusionModel, self).__init__()
        D = K * d
        self.fc17 = torch.nn.Linear(D, D)
        self.fc18 = torch.nn.Linear(D, D)
        self.fc19 = torch.nn.Linear(D, D)
        self.fc20 = torch.nn.Linear(D, D)
        self.sigmoid = torch.nn.Sigmoid()

        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.fc17.weight)
        nn.init.xavier_uniform_(self.fc18.weight)
        nn.init.xavier_uniform_(self.fc19.weight)
        nn.init.xavier_uniform_(self.fc20.weight)

    def forward(self, HS, HT):
        XS = self.fc17(HS)
        XT = self.fc18(HT)
        z = self.sigmoid(torch.add(XS, XT))
        H = torch.add((z * HS), ((1 - z) * HT))
        H = self.fc20(F.gelu(self.fc19(H)))
        return H

# Define a Spatial-Temporal Attention Model
class STAttModel(torch.nn.Module):
    def __init__(self, K, d, adj, device):
        super(STAttModel, self).__init__()
        D = K * d
        self.fc30 = torch.nn.Linear(7 * D, D)
        self.gcn = gcn(K, d)
        self.gcn1 = randomGAT(K, d, adj[0], device)
        self.gcn2 = randomGAT(K, d, adj[0], device)
        self.gcn3 = randomGAT(K, d, adj[1], device)
        self.gcn4 = randomGAT(K, d, adj[1], device)
        self.temporalAttention = TemporalAttentionModel(K, d, device)
        self.gatedFusion = GatedFusionModel(K, d)

        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.fc30.weight)

    def forward(self, X, STE, adp, Mask=True):
        HS1 = self.gcn1(X, STE)
        HS2 = self.gcn2(HS1, STE)
        HS3 = self.gcn3(X, STE)
        HS4 = self.gcn4(HS3, STE)
        HS5 = self.gcn(X, STE, adp)
        HS6 = self.gcn(HS5, STE, adp)
        HS = torch.cat((X, HS1, HS2, HS3, HS4, HS5, HS6), dim=-1)
        HS = F.gelu(self.fc30(HS))
        HT = self.temporalAttention(X, STE, Mask)
        H = self.gatedFusion(HS, HT)
        return torch.add(X, H)

# Define a Transform Attention Model
class TransformAttentionModel(torch.nn.Module):
    def __init__(self, K, d, dropout=0.3):
        super(TransformAttentionModel, self).__init__()
        D = K * d
        self.fc21 = torch.nn.Linear(D, D)
        self.fc22 = torch.nn.Linear(D, D)
        self.fc23 = torch.nn.Linear(D, D)
        self.fc24 = torch.nn.Linear(D, D)
        self.fc25 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = dropout

        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.fc21.weight)
        nn.init.xavier_uniform_(self.fc22.weight)
        nn.init.xavier_uniform_(self.fc23.weight)
        nn.init.xavier_uniform_(self.fc24.weight)
        nn.init.xavier_uniform_(self.fc25.weight)

    def forward(self, X, STE_P, STE_Q, mask1=False):
        query = F.gelu(self.fc21(STE_Q))
        key = F.gelu(self.fc22(STE_P))
        value = F.gelu(self.fc23(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, X.shape[0] // self.K, dim=0), dim=-1)
        X = self.fc25(F.gelu(self.fc24(X)))
        return X

# Define a Spatial-Temporal Model
class SpatialTemporalModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, K, d, SEDims, TEDims, P, L, device, adj, num_nodes):
        super(SpatialTemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rgdan = RGDAN(K, d, SEDims, TEDims, P, L, device, adj, num_nodes)

    def forward(self, x, se, te):
        lstm_out, _ = self.lstm(x)
        output = self.rgdan(lstm_out, se, te)
        return output

# Define a Residual Gated Dynamic Attention Network (RGDAN) ####
class RGDAN(torch.nn.Module):
    def __init__(self, K, d, SEDims, TEDims, P, L, device, adj, num_nodes):
        super(RGDAN, self).__init__()
        D = K * d
        self.fc1 = torch.nn.Linear(1, D)
        self.fc2 = torch.nn.Linear(D, D)
        self.STEmb = STEmbModel(SEDims, TEDims, K * d, device)
        self.STAttBlockEnc = STAttModel(K, d, adj, device)
        self.STAttBlockDec = STAttModel(K, d, adj, device)
        self.transformAttention = TransformAttentionModel(K, d)
        self.P = P
        self.L = L
        self.device = device
        self.fc26 = torch.nn.Linear(D, D)
        self.fc27 = torch.nn.Linear(D, 1)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.1)

        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc26.weight)
        nn.init.xavier_uniform_(self.fc27.weight)

    def forward(self, X, SE, TE):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        X = X.unsqueeze(3)
        X = self.fc2(F.gelu(self.fc1(X)))
        STE = self.STEmb(SE, TE)
        STE_P = STE[:, :self.P]
        STE_Q = STE[:, self.P:]
        X = self.STAttBlockEnc(X, STE_P, adp, Mask=True)
        X = self.transformAttention(X, STE_P, STE_Q)
        X = self.STAttBlockDec(X, STE_Q, adp, Mask=True)
        X = self.fc27(self.dropout(F.gelu(self.fc26(X))))
        return X.squeeze(3)

# Define loss functions
def mae_loss(pred, label, device):
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask != mask] = 0
    loss = torch.abs(pred - label)
    loss *= mask
    loss[loss != loss] = 0
    loss = torch.mean(loss)
    return loss

def mse_loss(pred, label, device):
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask != mask] = 0
    loss = (pred - label) ** 2
    loss *= mask
    loss[loss != loss] = 0
    loss = torch.mean(loss)
    return loss

def mape_loss(pred, label, device):
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask != mask] = 0
    loss = torch.abs(pred - label) / label
    loss *= mask
    loss[loss != loss] = 0
    loss = torch.mean(loss)
    return loss
