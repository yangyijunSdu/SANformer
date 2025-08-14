"""
SANformer for PEMFC
"""
import os
import math
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# ====================== Configuration ======================
class Config:
    # Main Control
    LOAD_MODEL = True  # True: load our trained model, False: train YOUR NEW model
    DATASET = 'FC1'  # 'FC1' or 'FC2'
    PRED_LEN = 1    # 1, 3, or 5 steps
    
    # Data Paths
    FC1_PATH = './data/FC1_Ageing_processed.csv'
    FC2_PATH = './data/FC2_Ageing_processed.csv'
    FC1_PRED_START = 578
    FC2_PRED_START = 509
    
    # Model Architecture
    if DATASET == 'FC1':
        SEQ_LEN = 24
    else:
        SEQ_LEN = 12
    LABEL_LEN = 1  
    ENC_IN = 24
    D_MODEL = 128
    N_HEADS = 2
    E_LAYERS = 2
    D_FF = 256
    DROPOUT = 0.11
    
    # Factor Learner
    P_HIDDEN_DIMS = [512, 512]
    P_HIDDEN_LAYERS = 2
    LSTM_HIDDEN_DIM = 128
    LSTM_LAYERS = 2
    
    # Training
    if DATASET == 'FC1':
        LEARNING_RATE = 0.0002
    else:
        LEARNING_RATE = 0.0008
    BATCH_SIZE = 50
    TRAIN_EPOCHS = 100
    PATIENCE = 200
    
    # System
    USE_GPU = torch.cuda.is_available()
    GPU_ID = 0
    CHECKPOINT_DIR = './checkpoints/'
    RESULTS_DIR = './results/'
    
    # Fixed
    TARGET = 'Utot'
    KERNEL_SIZE = 5
    ACTIVATION = 'relu'
    USE_NORM = True
    INVERSE = True
    NUM_WORKERS = 0

# ====================== Data Processing ======================
class FCDataset(Dataset):
    def __init__(self, data_path, mode='train', seq_len=24, pred_len=1, 
                 target='Utot', split_ratios=[0.5, 0.1, 0.4], 
                 pred_start_point=578, label_len=1):
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.target = target
        self.mode = mode
        self.pred_start_point = pred_start_point
        
        self.raw_data = pd.read_csv(data_path)
        self._prepare_data()
        self._split_data(split_ratios)
        
    def _prepare_data(self):
        columns = list(self.raw_data.columns)
        columns.remove(self.target)
        columns.remove('Time')
        
        self.raw_data = self.raw_data[['Time'] + columns + [self.target]]
        
        self.features = self.raw_data[columns].values
        self.target_values = self.raw_data[[self.target]].values
        self.timestamps = self.raw_data[['Time']].values
        
    def _split_data(self, ratios):
        total_len = len(self.raw_data)
        train_len = int(total_len * ratios[0])
        val_len = int(total_len * ratios[1])
        test_len = total_len - train_len - val_len
        
        boundaries = {
            'train': (0, train_len),
            'val': (train_len - self.seq_len, train_len + val_len),
            'test': (total_len - test_len - self.seq_len, total_len),
            'pred': (self.pred_start_point - self.seq_len, total_len)
        }
        
        start_idx, end_idx = boundaries[self.mode]
        
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.time_scaler = MinMaxScaler()
        
        train_features = self.features[:train_len]
        train_target = self.target_values[:train_len]
        
        self.feature_scaler.fit(train_features)
        self.target_scaler.fit(train_target)
        self.time_scaler.fit(self.timestamps)
        
        features_scaled = self.feature_scaler.transform(self.features)
        target_scaled = self.target_scaler.transform(self.target_values)
        time_scaled = self.time_scaler.transform(self.timestamps[start_idx:end_idx])
        
        self.data_x = np.concatenate([features_scaled, target_scaled], axis=1)[start_idx:end_idx]
        self.data_y = np.concatenate([features_scaled, target_scaled], axis=1)[start_idx:end_idx]
        self.data_stamp = time_scaled.astype(np.float32)
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return (torch.FloatTensor(seq_x), 
                torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_x_mark), 
                torch.FloatTensor(seq_y_mark))
    
    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)

# ====================== Model Components ======================
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * 
                   -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size
    
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        
        moving_mean = self.moving_avg(x_pad.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        
        seasonal = x - moving_mean
        return seasonal, moving_mean

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='m', dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)

class Projector(nn.Module):
    def __init__(self, seq_len, hidden_dims, hidden_layers, output_dim, 
                 lstm_hidden, lstm_layers):
        super().__init__()
        
        self.gru_x = nn.GRU(
            input_size=seq_len,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.gru_stats = nn.GRU(
            input_size=seq_len,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        
        mlp_input_dim = 3 * lstm_hidden
        layers = [nn.Linear(mlp_input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, x, stats):
        _, h_x = self.gru_x(x)
        h_forward = h_x[-2]
        h_backward = h_x[-1]
        x_features = torch.cat([h_forward, h_backward], dim=1)
        
        _, h_stats = self.gru_stats(stats)
        stats_features = h_stats[-1]
        
        combined = torch.cat([x_features, stats_features], dim=1)
        return self.backbone(combined)

class DSAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, 
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        if tau is not None:
            tau = tau.unsqueeze(1).unsqueeze(1)
            scores = scores * tau
        if delta is not None:
            delta = delta.unsqueeze(1).unsqueeze(1)
            scores = scores + delta
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
        out = out.view(B, L, -1)
        
        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for i, attn_layer in enumerate(self.attn_layers):
            delta_i = delta if i == 0 else None
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta_i)
            attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns

# ====================== Main Model ======================
class PredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.SEQ_LEN
        self.pred_len = config.PRED_LEN
        self.use_norm = config.USE_NORM
        self.d_model = config.D_MODEL
        self.enc_in = config.ENC_IN
        self.individual = False
        
        self.decomposition = SeriesDecomposition(config.KERNEL_SIZE)
        
        self.enc_embedding = DataEmbedding_inverted(
            c_in=config.SEQ_LEN,
            d_model=config.D_MODEL,
            embed_type='timeF',
            freq='m',
            dropout=config.DROPOUT
        )
        
        self.tau_learner = Projector(
            seq_len=config.SEQ_LEN,
            hidden_dims=config.P_HIDDEN_DIMS,
            hidden_layers=config.P_HIDDEN_LAYERS,
            output_dim=1,
            lstm_hidden=config.LSTM_HIDDEN_DIM,
            lstm_layers=config.LSTM_LAYERS
        )
        
        self.delta_learner = Projector(
            seq_len=config.SEQ_LEN,
            hidden_dims=config.P_HIDDEN_DIMS,
            hidden_layers=config.P_HIDDEN_LAYERS,
            output_dim=config.ENC_IN + 1,
            lstm_hidden=config.LSTM_HIDDEN_DIM,
            lstm_layers=config.LSTM_LAYERS
        )
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, attention_dropout=config.DROPOUT, output_attention=False),
                        config.D_MODEL,
                        config.N_HEADS
                    ),
                    config.D_MODEL,
                    config.D_FF,
                    dropout=config.DROPOUT,
                    activation=config.ACTIVATION
                ) for _ in range(config.E_LAYERS)
            ],
            norm_layer=nn.LayerNorm(config.D_MODEL)
        )
        
        self.seasonal_linear = nn.Linear(config.SEQ_LEN, config.D_MODEL)
        self.seasonal_linear.weight = nn.Parameter((1/config.SEQ_LEN) * torch.ones([config.D_MODEL, config.SEQ_LEN]))
        
        self.projector = nn.Linear(config.D_MODEL, config.PRED_LEN, bias=True)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()
        x_mark_enc_raw = x_mark_enc.clone().detach()
        x_raw = torch.cat([x_raw, x_mark_enc_raw], dim=2)
        xlinear = x_raw.clone().detach()
        
        x_raw = x_raw.permute(0, 2, 1)
        means_inverted = x_raw.mean(1, keepdim=True).detach()
        stdev_inverted = torch.sqrt(torch.var(x_raw, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        
        seasonal_init, trend_init = self.decomposition(xlinear)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        _, _, N = x_enc.shape
        
        tau = self.tau_learner(x_raw, stdev_inverted).exp()
        delta = self.delta_learner(x_raw, means_inverted)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                         dtype=seasonal_init.dtype).to(seasonal_init.device)
            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.seasonal_linear[i](seasonal_init[:, i, :])
        else:
            seasonal_output = self.seasonal_linear(seasonal_init)
        
        enc_out = enc_out + seasonal_output
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out, attns
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]

# ====================== Training System ======================
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(f'cuda:{config.GPU_ID}' if config.USE_GPU else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = []
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            dec_inp = torch.zeros_like(batch_y[:, -self.config.PRED_LEN:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.config.LABEL_LEN, :], dec_inp], dim=1).float().to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            outputs = outputs[:, -self.config.PRED_LEN:, -1:]
            batch_y = batch_y[:, -self.config.PRED_LEN:, -1:].to(self.device)
            
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss.append(loss.item())
        
        return np.mean(total_loss)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.config.PRED_LEN:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config.LABEL_LEN, :], dec_inp], dim=1).float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.config.PRED_LEN:, -1:]
                batch_y = batch_y[:, -self.config.PRED_LEN:, -1:].to(self.device)
                
                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.item())
        
        return np.mean(total_loss)
    
    def train(self, train_loader, val_loader, save_path):
        print("\nStarting training...")
        
        for epoch in range(self.config.TRAIN_EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if epoch % 100 == 0 and epoch > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.TRAIN_EPOCHS} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.model.load_state_dict(torch.load(save_path))
        print("Training completed!\n")
    
    def predict(self, dataloader, dataset, save_stride=None):
        self.model.eval()
        preds = []
        trues = []
        if save_stride is None:
            save_stride = self.config.PRED_LEN

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
                # only save every `save_stride` samples to avoid dense overlap
                if i % save_stride != 0:
                    continue

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.config.PRED_LEN:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config.LABEL_LEN, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.config.PRED_LEN:, -1:]
                batch_y = batch_y[:, -self.config.PRED_LEN:, -1:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                if self.config.INVERSE:
                    pred = dataset.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
                    true = dataset.inverse_transform(true.reshape(-1, 1)).reshape(true.shape)

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0) if len(preds) > 0 else np.empty((0,))
        trues = np.concatenate(trues, axis=0) if len(trues) > 0 else np.empty((0,))
        return preds, trues


# ====================== Utility Functions ======================
def compute_metrics(predictions, ground_truth):
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    mape = np.mean(np.abs((predictions - ground_truth) / (ground_truth + 1e-8))) * 100
    r2 = r2_score(ground_truth.flatten(), predictions.flatten())
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def save_prediction_results(predictions, ground_truth, metrics, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    pred_flat = predictions.flatten()
    true_flat = ground_truth.flatten()
    
    # Save CSV
    df = pd.DataFrame({
        'Time_Step': range(len(pred_flat)),
        'True_Value': true_flat,
        'Predicted_Value': pred_flat,
        'Error': pred_flat - true_flat
    })
    csv_path = os.path.join(save_path, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} data points to CSV")
    
    # Create visualization
    plt.figure(figsize=(20, 8))
    
    if len(pred_flat) <= 1000:
        plt.subplot(2, 1, 1)
        plt.plot(true_flat, label='True Values', color='blue', linewidth=1.5, alpha=0.8)
        plt.plot(pred_flat, label='Predictions', color='red', linewidth=1.5, alpha=0.8)
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage (V)')
        plt.title(f'Forecasting Results - All {len(pred_flat)} Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(pred_flat - true_flat, color='green', linewidth=1, alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Prediction Error')
        plt.title('Prediction Error Over Time')
        plt.grid(True, alpha=0.3)
    else:
        plt.subplot(2, 2, 1)
        plt.plot(true_flat, label='True Values', color='blue', linewidth=0.5, alpha=0.6)
        plt.plot(pred_flat, label='Predictions', color='red', linewidth=0.5, alpha=0.6)
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage (V)')
        plt.title(f'Complete Overview - {len(pred_flat)} Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        end_idx = min(500, len(pred_flat))
        plt.plot(true_flat[:end_idx], label='True Values', color='blue', linewidth=1.5)
        plt.plot(pred_flat[:end_idx], label='Predictions', color='red', linewidth=1.5)
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage (V)')
        plt.title(f'First {end_idx} Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        mid_start = len(pred_flat) // 2 - 250
        mid_end = min(mid_start + 500, len(pred_flat))
        x_axis = range(mid_start, mid_end)
        plt.plot(x_axis, true_flat[mid_start:mid_end], label='True Values', color='blue', linewidth=1.5)
        plt.plot(x_axis, pred_flat[mid_start:mid_end], label='Predictions', color='red', linewidth=1.5)
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage (V)')
        plt.title(f'Middle Section ({mid_start}-{mid_end})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        errors = pred_flat - true_flat
        plt.hist(errors, bins=50, color='green', alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'predictions_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        f.write("Performance Metrics\n")
        f.write("="*20 + "\n")
        for key, value in metrics.items():
            f.write(f'{key}: {value:.6f}\n')
        f.write(f'\nTotal Predictions: {len(pred_flat)}\n')
    
    return csv_path, plot_path

# ====================== Main Execution ======================
def main():
    config = Config()
    
    if config.USE_GPU:
        torch.cuda.set_device(config.GPU_ID)
    
    print("="*60)
    print("PEMFC DEGRADATION FORECASTING SYSTEM")
    print(f"Prediction Horizon: {config.PRED_LEN} steps")
    print("="*60)
    
    if config.LOAD_MODEL:
        print(f"\n*** LOAD MODE - Using Existing Model ***")
    else:
        print(f"\n*** TRAIN MODE - Training New Model ***")
    
    # Dataset selection
    if config.DATASET == 'FC1':
        data_path = config.FC1_PATH
        pred_start = config.FC1_PRED_START
    else:
        data_path = config.FC2_PATH
        pred_start = config.FC2_PRED_START
    
    print(f"Dataset: {config.DATASET}")
    print(f"Prediction Start: {pred_start}")
    
    # Model checkpoint with prediction length
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 
                                  f'{config.DATASET}_pred{config.PRED_LEN}_model.pth')
    
    if config.LOAD_MODEL and not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Model not found: {checkpoint_path}")
        print(f"Please train model for {config.PRED_LEN}-step prediction first.")
        return
    
    # Prepare datasets
    print("\nPreparing datasets...")
    test_dataset = FCDataset(data_path, mode='test', seq_len=config.SEQ_LEN,
                            pred_len=config.PRED_LEN, target=config.TARGET,
                            pred_start_point=pred_start, label_len=config.LABEL_LEN)
    
    pred_dataset = FCDataset(data_path, mode='pred', seq_len=config.SEQ_LEN,
                            pred_len=config.PRED_LEN, target=config.TARGET,
                            pred_start_point=pred_start, label_len=config.LABEL_LEN)
    
    test_loader = DataLoader(test_dataset, batch_size=1, 
                           shuffle=False, num_workers=config.NUM_WORKERS, drop_last=False)
    pred_loader = DataLoader(pred_dataset, batch_size=1, 
                           shuffle=False, num_workers=config.NUM_WORKERS, drop_last=False)
    
    if not config.LOAD_MODEL:
        train_dataset = FCDataset(data_path, mode='train', seq_len=config.SEQ_LEN,
                                 pred_len=config.PRED_LEN, target=config.TARGET,
                                 pred_start_point=pred_start, label_len=config.LABEL_LEN)
        
        val_dataset = FCDataset(data_path, mode='val', seq_len=config.SEQ_LEN,
                               pred_len=config.PRED_LEN, target=config.TARGET,
                               pred_start_point=pred_start, label_len=config.LABEL_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, 
                              shuffle=False, num_workers=config.NUM_WORKERS, drop_last=True)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    print(f"Test: {len(test_dataset)}, Pred: {len(pred_dataset)}")
    
    # Initialize model and trainer
    model = PredictionModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, config)
    
    # Train or load model
    if config.LOAD_MODEL:
        print(f"Loading model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        model.load_state_dict(checkpoint)
        model.to(trainer.device)
        print("Model loaded successfully!")
    else:
        trainer.train(train_loader, val_loader, checkpoint_path)
        print(f"Model saved: {checkpoint_path}")
    
    # Test
    print("\nTesting...")
    test_preds, test_trues = trainer.predict(test_loader, test_dataset)
    test_metrics = compute_metrics(test_preds, test_trues)
    
    print("Test Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Prediction
    print("\nPredicting...")
    pred_preds, pred_trues = trainer.predict(pred_loader, pred_dataset)
    pred_metrics = compute_metrics(pred_preds, pred_trues)
    
    print("Prediction Results:")
    for key, value in pred_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Save results with prediction length
    save_path = os.path.join(config.RESULTS_DIR, 
                            f'{config.DATASET}_pred{config.PRED_LEN}')
    if config.LOAD_MODEL:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'{save_path}_{timestamp}'
    
    csv_file, plot_file = save_prediction_results(pred_preds, pred_trues, pred_metrics, save_path)
    
    print(f"\nResults saved:")
    print(f"  CSV: {csv_file}")
    print(f"  Plot: {plot_file}")
    
    print("\n" + "="*60)
    print(f"COMPLETED - {config.PRED_LEN}-step prediction")
    print("="*60)

if __name__ == '__main__':
    main()