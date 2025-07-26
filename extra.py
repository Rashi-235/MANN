import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.nn import HuberLoss
from sklearn.preprocessing import MinMaxScaler


def nse(true, pred):
    # Nash–Sutcliffe Efficiency: 1 - (Σ(e²)/Σ((t−t̄)²))
    num = ((true - pred)**2).sum(axis=0)
    den = ((true - true.mean(axis=0))**2).sum(axis=0)
    return 1 - num/den

def rsr(true, pred):
    # Ratio of RMSE to standard deviation
    rmse = np.sqrt(((true - pred)**2).mean(axis=0))
    std  = np.std(true, axis=0)
    return rmse / std

def pbias(true, pred):
    # Percent bias: 100 * Σ(pred - true) / Σ(true)
    return 100.0 * ( (pred - true).sum(axis=0) / true.sum(axis=0) )

def tpe(true, pred, eps=0.01):
    # Clamp denominator so it never goes below eps
    pct_err = np.abs((pred - true) / np.maximum(true, eps)) * 100
    k = max(1, int(0.05 * len(pct_err)))
    return np.sort(pct_err, axis=0)[-k:].mean(axis=0)



class T2V(nn.Module):
    """
    Time-to-Vector embedding replacing fixed positional encoding.
    Produces a learned sinusoidal embedding of a scalar time index.
    """
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.W = None
        self.P = None

    def build(self, seq_len: int):
        # W: maps scalar to output_dim features
        self.W = nn.Parameter(torch.empty(1, self.output_dim))
        # P: per-position phase offset [seq_len, output_dim]
        self.P = nn.Parameter(torch.empty(seq_len, self.output_dim))
        nn.init.uniform_(self.W)
        nn.init.uniform_(self.P)

    def forward(self, x: torch.Tensor):
        # x: [B, L, 1] scalar time index
        B, L, _ = x.shape
        if self.W is None:
            self.build(L)
        # x @ W: [B, L, output_dim]
        proj = x @ self.W  # broadcast multiplication
        # add phase P: [L, output_dim] -> broadcast to [B, L, output_dim]
        sin_in = proj + self.P.unsqueeze(0)
        return torch.sin(sin_in)


class MemoryBank(nn.Module):
    """
    External memory bank for storing hydrological patterns 
    """
    def __init__(self, memory_size, memory_dim, num_heads=4):
        super(MemoryBank, self).__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        # Initialize memory bank
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.register_buffer('usage', torch.zeros(memory_size))
        
        # Memory interaction layers
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        
    def read(self, query, top_k=5):
        batch_size = query.size(0)
        q = self.query_proj(query)
        k = self.key_proj(self.memory)
        scores = torch.matmul(q, k.transpose(0, 1))
        attention_weights = F.softmax(scores / math.sqrt(self.memory_dim), dim=-1)
        v = self.value_proj(self.memory)
        read_content = torch.matmul(attention_weights, v)
        return read_content, attention_weights
    
    def write(self, content, gate=None):
        if gate is None:
            gate = torch.sigmoid(torch.randn(1, device=content.device))
        _, least_used_idx = torch.min(self.usage, dim=0)
        # with torch.no_grad():
        self.memory.data[least_used_idx] = (
                gate * content + (1 - gate) * self.memory.data[least_used_idx]
            )
        self.usage[least_used_idx] += 1
        return least_used_idx


class MANNTransformerEncoder(nn.Module):
    """
    Memory-Augmented Transformer Encoder with learned T2V embedding
    """
    def __init__(
        self, d_model, nhead, num_layers, memory_size, memory_dim, 
        dropout=0.1, activation='relu'
    ):
        super(MANNTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Memory bank
        self.memory_bank = MemoryBank(memory_size, memory_dim, nhead)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, 
            dropout=dropout, activation=activation, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Memory integration layers
        self.memory_gate = nn.Linear(d_model + memory_dim, d_model)
        self.memory_proj = nn.Linear(d_model, memory_dim)
        
        # Learned time-to-vector embedding
        self.t2v = T2V(output_dim=d_model)
        
    def forward(self, src, src_mask=None):
        """
        src: [batch_size, seq_len, d_model]
        """
        B, L, D = src.size()
        
        # Build position indices 0..L-1 and expand
        pos = torch.arange(L, device=src.device, dtype=src.dtype)
        pos = pos.view(1, L, 1).expand(B, L, 1)
        # Learned sinusoidal embedding
        time_embed = self.t2v(pos)  # [B, L, d_model]
        
        # Add the learned positional embedding
        src = src + time_embed
        
        # Transformer encoding
        encoded = self.transformer_encoder(src, src_mask)
        sequence_repr = encoded.mean(dim=1)  # [B, D]

    # Project to memory query space
        mem_q = self.memory_proj(sequence_repr)  # [B, memory_dim]
        mem_c, attn = self.memory_bank.read(mem_q)  # [B, memory_dim], [B, memory_size]

    # Optionally, concatenate memory content to each timestep
        mem_c_expanded = mem_c.unsqueeze(1).expand(-1, L, -1)  # [B, L, memory_dim]
        combined = torch.cat([encoded, mem_c_expanded], dim=-1)  # [B, L, D + memory_dim]
        memory_enhanced = self.memory_gate(combined)
        return memory_enhanced, attn



class DischargePredictor(nn.Module):
    """
    Complete MANN-Transformer model for discharge prediction
    """
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6, 
                 memory_size=100, memory_dim=128, output_dim=1, dropout=0.1):
        super(DischargePredictor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # MANN-Transformer encoder
        self.mann_transformer = MANNTransformerEncoder(
            d_model, nhead, num_layers, memory_size, memory_dim, dropout
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(d_model, output_dim)
        
    def forward(self, x, return_attention=False):
        """
        Forward pass for discharge prediction
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to model dimension
        x_projected = self.input_projection(x)
        
        # MANN-Transformer encoding
        encoded, attention_weights = self.mann_transformer(x_projected)
        
        # Use final timestep for prediction
        final_state = encoded[:, -1, :]  # [batch_size, d_model]
        
        # Discharge prediction
        discharge_pred = self.output_projection(final_state)
        
        # Uncertainty estimation
        uncertainty = torch.exp(self.uncertainty_head(final_state))
        
        if return_attention:
            return discharge_pred, uncertainty, attention_weights
        return discharge_pred, uncertainty
     
     

class HydrologicalDataset(Dataset):
    """
    Dataset class for hydrological time series with few-shot learning support
    Now returns (features, target, target_date).
    """
    def __init__(self, data, sequence_length=30, prediction_horizon=1, 
                 support_size=5, query_size=15):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.support_size = support_size
        self.query_size = query_size
        
        # Prepare features and targets
        self.prepare_data()
        
    def prepare_data(self):
        """
        Prepare time series data with seasonal and trend features
        """
        # Extract date features
        self.data['day_of_year'] = self.data.index.dayofyear / 365.0
        self.data['month'] = self.data.index.month / 12.0
        self.data['season_sin'] = np.sin(2 * np.pi * self.data.index.dayofyear / 365.0)
        self.data['season_cos'] = np.cos(2 * np.pi * self.data.index.dayofyear / 365.0)
        
        # Lag features
        for lag in [1, 7, 30, 365]:
            self.data[f'discharge_lag_{lag}'] = self.data['Simga'].shift(lag)
        
        # Rolling statistics
        for window in [7, 30]:
            self.data[f'discharge_mean_{window}'] = (
                self.data['Simga'].rolling(window).mean()
            )
            self.data[f'discharge_std_{window}'] = (
                self.data['Simga'].rolling(window).std()
            )
        # Feature columns
        self.feature_cols = [c for c in self.data.columns 
                     if c not in ('Simga','Simga_scaled')]


        for col in self.feature_cols:
            col_min = self.data[col].min()
            col_max = self.data[col].max()
            self.data[col] = (self.data[col] - col_min) / (col_max - col_min)
        
        # Remove rows with NaN
        self.data = self.data.dropna()
        
        
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        start = idx
        end   = start + self.sequence_length

        # 1) grab the input window
        X = self.data[self.feature_cols].iloc[start:end].values

        # 2) grab the next prediction_horizon steps as a vector
        tgt_start = end
        tgt_end   = end + self.prediction_horizon
        Y = self.data['Simga_scaled'].iloc[tgt_start:tgt_end].values

        # 3) grab the corresponding dates if you still want them
        dates = list(self.data.index[tgt_start:tgt_end])

        return (
            torch.FloatTensor(X),             # (sequence_length, num_features)
            torch.FloatTensor(Y),             # (prediction_horizon,)
            dates                             # list of Timestamps, length prediction_horizon
        )

    
    def get_few_shot_batch(self, num_tasks=1):
        """
        Generate few-shot tasks: each task returns support_x, support_y, support_dates,
        and query_x, query_y, query_dates.
        """
        tasks = []
        for _ in range(num_tasks):
            total_needed = self.support_size + self.query_size
            if len(self) < total_needed:
                continue
                
            indices = np.random.choice(len(self), total_needed, replace=False)
            support_indices = indices[:self.support_size]
            query_indices   = indices[self.support_size:]
            
            # Support set
            support_x, support_y, support_dates = [], [], []
            for i in support_indices:
                x, y, dt = self[i]
                support_x.append(x)
                support_y.append(y)
                support_dates.append(dt)
            
            # Query set
            query_x, query_y, query_dates = [], [], []
            for i in query_indices:
                x, y, dt = self[i]
                query_x.append(x)
                query_y.append(y)
                query_dates.append(dt)
            
            tasks.append({
                'support_x'     : torch.stack(support_x),
                'support_y'     : torch.stack(support_y),
                'support_dates' : support_dates,
                'query_x'       : torch.stack(query_x),
                'query_y'       : torch.stack(query_y),
                'query_dates'   : query_dates
            })
        
        return tasks


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()

def plot_true_vs_pred(true_vals, pred_vals):
    plt.figure(figsize=(8, 8))
    plt.scatter(true_vals, pred_vals, alpha=0.5)
    plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')
    plt.xlabel('True Discharge (m³/s)')
    plt.ylabel('Predicted Discharge (m³/s)')
    plt.title('True vs Predicted Discharge')
    plt.grid(True)
    plt.savefig('true_vs_pred.png')
    plt.close()


def plot_observed_vs_predicted(dates, observed, predicted):
    plt.figure(figsize=(15, 8))
    
    # Convert dates to proper format if needed
    if isinstance(dates[0], (int, float)):
        # If dates are indices, create a simple range
        x_axis = range(len(dates))
        xlabel = 'Time (Days)'
    else:
        # If dates are actual dates
        x_axis = dates
        xlabel = 'Date'
    
    # Plot observed and predicted as clean lines
    plt.plot(x_axis, observed, 'b-', label='Observed', linewidth=2, alpha=0.8)
    plt.plot(x_axis, predicted, 'r-', label='Predicted', linewidth=2, alpha=0.8)
    
    # Calculate confidence interval (optional)
    residuals = np.array(predicted) - np.array(observed)
    std_residual = np.std(residuals)
    upper_bound = np.array(predicted) + 1.96 * std_residual
    lower_bound = np.array(predicted) - 1.96 * std_residual
    
    # Add confidence interval as shaded area
    plt.fill_between(x_axis, lower_bound, upper_bound, 
                     color='gray', alpha=0.3, label='Confidence Interval')
    
    plt.xlabel(xlabel)
    plt.ylabel('Inflow (m³/s)')
    plt.title('Observed vs Predicted Discharge Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis if using actual dates
    if not isinstance(dates[0], (int, float)):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('obs_vs_pred_time.png', dpi=300, bbox_inches='tight')
    plt.close()

# Also add a function to limit data points for cleaner visualization
def plot_observed_vs_predicted_limited(dates, observed, predicted, max_points=300):
    """Plot with limited points for cleaner visualization"""
    
    # Limit to max_points for cleaner plot
    if len(observed) > max_points:
        step = len(observed) // max_points
        dates = dates[::step]
        observed = observed[::step]
        predicted = predicted[::step]
    
    plt.figure(figsize=(15, 8))
    
    # Create time axis
    x_axis = range(len(dates))
    
    # Plot as clean lines
    plt.plot(x_axis, observed, 'b-', label='Observed', linewidth=2.5, alpha=0.9)
    plt.plot(x_axis, predicted, 'r-', label='Predicted', linewidth=2.5, alpha=0.9)
    
    # Calculate and plot confidence interval
    residuals = np.array(predicted) - np.array(observed)
    std_residual = np.std(residuals)
    upper_bound = np.array(predicted) + 1.96 * std_residual
    lower_bound = np.array(predicted) - 1.96 * std_residual
    
    plt.fill_between(x_axis, lower_bound, upper_bound, 
                     color='gray', alpha=0.3, label='Confidence Interval')
    
    plt.xlabel('Time (Days)')
    plt.ylabel('Inflow (m³/s)')
    plt.title('Observed vs Predicted Discharge Over Time')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('obs_vs_pred_time_clean.png', dpi=300, bbox_inches='tight')
    plt.close()

def compute_rmse(model, dataset, device='cpu'):
    model.eval()
    mse_sum = 0.0
    count = 0
    with torch.no_grad():
        for features, target, _ in dataset:
            # features: (seq_len, feat), target: (horizon,)
            x = features.unsqueeze(0).to(device)   # [1, L, feat]
            y_true = target.to(device)             # [horizon]
            y_pred, _ = model(x)                   # [1, horizon]
            y_pred = y_pred.squeeze(0)             # [horizon]
            
            err = y_pred - y_true                  # [horizon]
            mse_sum += torch.sum(err * err).item()
            count   += err.numel()
    rmse = math.sqrt(mse_sum / count) if count>0 else float('nan')
    return rmse



def train_mann_transformer(model, dataset, num_epochs=50, batch_size=32, 
                          learning_rate=1e-5, device='cpu', patience=5, min_delta=1e-4):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    huber_loss = nn.HuberLoss(delta=1.0)
    
    # Create validation split (20% of data)
    total = len(dataset)
    test_size = val_size = int(0.15 * total)
    train_size = total - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    memory_write_frequency = 10
    
    # For final plots
    all_val_dates = []
    all_val_true = []
    all_val_pred = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        few_shot_batches = dataset.get_few_shot_batch(num_tasks=batch_size)
        for batch_idx, task in enumerate(few_shot_batches):
            optimizer.zero_grad()
            
            # Support set processing
            support_x = task['support_x'].to(device)
            support_y = task['support_y'].to(device)
            _, _, support_attention = model(support_x, return_attention=True)
            
            if batch_idx % memory_write_frequency == 0:
                with torch.no_grad():
                    support_repr = torch.mean(support_attention, dim=0)
                    for i in range(min(support_repr.size(0), 5)):
                        model.mann_transformer.memory_bank.write(support_repr[i])
            
            # Query set processing
            query_x = task['query_x'].to(device)
            query_y = task['query_y'].to(device)
            query_dates = task['query_dates']
            
            query_pred, query_uncertainty = model(query_x)
            
            # Loss calculation
            support_loss = huber_loss(_, _.detach())
            query_loss = huber_loss(query_pred.squeeze(), query_y.squeeze())
            uncertainty_loss = torch.mean(query_uncertainty)
            total_loss = support_loss + query_loss + 0.1 * uncertainty_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            # Convert to raw discharge
            true_log1p = query_y.squeeze().detach().cpu().numpy()
            pred_log1p = query_pred.squeeze().detach().cpu().numpy()
            true_raw = np.expm1(true_log1p).tolist()   # e.g. [12.3, 11.5, 10.2, 9.8, 9.1]
            pred_raw = np.expm1(pred_log1p).tolist()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, target, date in val_dataset:
                features = features.unsqueeze(0).to(device)
                target   = target.to(device)            # shape [5]
                pred, _  = model(features)              # shape [1,5]
                loss     = huber_loss(pred.squeeze(0), target)

                val_loss += loss.item()

        # vector → raw
                true_raw = np.expm1(target.cpu().numpy()).tolist()  # [t+1, t+2, …]
                pred_raw = np.expm1(pred.squeeze(0).cpu().numpy()).tolist()

        # keep only t+1
                all_val_dates.append(date[0])
                all_val_true.append(true_raw[0])
                all_val_pred.append(pred_raw[0])
        
        avg_val_loss = val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)
        
        # Record training loss
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        train_losses.append(avg_epoch_loss)
        
        scheduler.step()
        print(f"Epoch {epoch} complete. Train Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}\n")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}: no improvement in validation loss for {patience} epochs.")
            early_stop = True
            break


    test_rmse = compute_rmse(model, test_dataset, device)
    print(f"Overall Test RMSE: {test_rmse:.4f}")
    
     # ——— STEP‑WISE HORIZON METRICS ———
    all_true_vectors = []
    all_pred_vectors = []
    model.eval()
    with torch.no_grad():
        for features, target, _ in test_dataset:
            # features: (30,5), target: (5,)
            x = features.unsqueeze(0).to(device)  # (1,30,5)
            pred_log1p, _ = model(x)              # (1,5)
            # true_log1p = target.cpu().numpy()     # (5,)
            # pred_log1p = pred_log1p.cpu().numpy().squeeze()  # (5,)
            pred_scaled = pred.squeeze(0).cpu().numpy().reshape(-1,1)
            pred_log1p  = target_scaler.inverse_transform(pred_scaled).flatten()
            all_pred_vectors.append(np.expm1(pred_log1p))
            true_scaled = target.cpu().numpy().reshape(-1,1)
            true_log1p  = target_scaler.inverse_transform(true_scaled).flatten()
            all_true_vectors.append(np.expm1(true_log1p))
            # back to raw discharge
            # all_true_vectors.append(np.expm1(true_log1p))
            # all_pred_vectors.append(np.expm1(pred_log1p))

    all_true = np.stack(all_true_vectors)  # (N,5)
    all_pred = np.stack(all_pred_vectors)  # (N,5)

    # compute and print your 5‐step metrics
    print("NSE Values    :", np.round(nse(all_true,   all_pred),   4))
    print("RSR Values    :", np.round(rsr(all_true,   all_pred),   4))
    print("PBIAS (%)     :", np.round(pbias(all_true, all_pred),   4))
    print("TPE (%)       :", np.round(tpe(all_true,   all_pred),   4))


    # Generate plots after training
    plot_losses(train_losses, val_losses)
    plot_true_vs_pred(all_val_true, all_val_pred)
    plot_observed_vs_predicted_limited(all_val_dates, all_val_true, all_val_pred, max_points=300)
    
    return train_losses, val_losses



def prepare_discharge_data(file_path):
    """
    Prepare the discharge data for MANN-Transformer training
    (now dropping any rows where 'Simga' is NaN)
    """
    # 1. Load
    print("--- 1. Loading and Preparing Data for Prediction ---")
    data = pd.read_excel(file_path)
    print("Successfully loaded data from Excel file.")

    # 2. Check missing
    missing_before = data.isna().sum().sum()
    print(f"Missing values before imputation: {missing_before}")

    # (if you do any imputation, do it here…)
    # e.g. data.fillna(method="ffill", inplace=True)
    missing_after = data.isna().sum().sum()
    print(f"Missing values after imputation: {missing_after}\n")

    # 3. Index and log‑transform
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    data.dropna(subset=['Simga'], inplace=True)
    data['Simga'] = np.log1p(data['Simga'])

    target_scaler = MinMaxScaler()
    data['Simga_scaled'] = target_scaler.fit_transform(data[['Simga']])

    return data, target_scaler


# Main execution with error handling
if __name__ == "__main__":
        # Initialize model and dataset
        # data = prepare_discharge_data('DOC-20250406-WA0050..xlsx')
        df, target_scaler = prepare_discharge_data("Simga.xlsx")

    # — split, smooth, scale —
        print("--- 2. Splitting, Smoothing, and Scaling Data ---")
        train_df = df['2000-06-01':'2010-09-30'].copy()
        test_df  = df['2011-06-01':'2014-09-30'].copy()
        print(f"Training data shape: {train_df.shape} "
          f"(from {train_df.index.min()} to {train_df.index.max()})")
        print(f"Testing data shape:  {test_df.shape} "
          f"(from {test_df.index.min()} to {test_df.index.max()})")

    # now scaling
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        train_vals = scaler.fit_transform(train_df[['Simga']])
        test_vals  = scaler.transform(test_df [['Simga']])
        train_df['scaled'] = train_vals
        test_df ['scaled'] = test_vals
        print("Data scaling complete.\n")

        horizon = 5
        for h in range(1, horizon+1):
            train_df[f"y_t+{h}"] = train_df['scaled'].shift(-h)
            test_df [f"y_t+{h}"] = test_df ['scaled'].shift(-h)

# now drop the trailing rows with NaNs in targets
        train_df = train_df.iloc[:-horizon].copy()
        test_df  = test_df.iloc[:-horizon].copy()

        print("Multi‐step train shape:", train_df[[f"y_t+{h}" for h in range(1,horizon+1)]].shape)
        print("Multi‐step test  shape:", test_df [[f"y_t+{h}" for h in range(1,horizon+1)]].shape)


        dataset = HydrologicalDataset(df, sequence_length=30, prediction_horizon=5)
        
        # Model configuration
        input_dim = len(dataset.feature_cols)
        model = DischargePredictor(
            input_dim=input_dim,
            d_model=256,
            nhead=8,
            num_layers=6,
            memory_size=100,
            memory_dim=128,
            output_dim=5
        )
        
        # Automatic device detection
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Training with fixed implementation
        train_losses, val_losses = train_mann_transformer(model, dataset, num_epochs=50, device=device)
    
        print("Training completed successfully!")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        
        
   
