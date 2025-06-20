import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import math
from torch.nn import HuberLoss


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
        with torch.no_grad():
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
        
        # Memory interaction
        memory_enhanced = []
        for t in range(L):
            current = encoded[:, t, :]
            mem_q = self.memory_proj(current)
            mem_c, attn = self.memory_bank.read(mem_q)
            combined = torch.cat([current, mem_c], dim=-1)
            enhanced = self.memory_gate(combined)
            memory_enhanced.append(enhanced)
        
        memory_enhanced = torch.stack(memory_enhanced, dim=1)
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
    

    # Uncertainty estimation
        uncertainty = torch.exp(self.uncertainty_head(final_state))

        if return_attention:
            return discharge_pred, uncertainty, attention_weights
        return discharge_pred, uncertainty

    # def forward(self, x, return_attention=False):
    #     batch_size, seq_len, _ = x.size()
    #     x_projected = self.input_projection(x)
    #     encoded, attention_weights = self.mann_transformer(x_projected)
    #     final_state = encoded[:, -1, :]

    # # Output projection with activation monitoring
    #     x_out = final_state
    #     zero_fracs = []
    #     neg_fracs = []
    #     for module in self.output_projection:
    #         x_out = module(x_out)
    #         if isinstance(module, nn.LeakyReLU):
    #             zero_frac = (x_out == 0).float().mean().item()
    #             neg_frac = (x_out < 0).float().mean().item()
    #             zero_fracs.append(zero_frac)
    #             neg_fracs.append(neg_frac)
    #             print(f"LeakyReLU: Zero activations={zero_frac:.4f}, Negative activations={neg_frac:.4f}")

    #     discharge_pred = x_out
    #     uncertainty = torch.exp(self.uncertainty_head(final_state))

    #     if return_attention:
    #         return discharge_pred, uncertainty, attention_weights
    #     return discharge_pred, uncertainty

  


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
            self.data[f'discharge_lag_{lag}'] = self.data['Discharge(m3/s)'].shift(lag)
        
        # Rolling statistics
        for window in [7, 30]:
            self.data[f'discharge_mean_{window}'] = (
                self.data['Discharge(m3/s)'].rolling(window).mean()
            )
            self.data[f'discharge_std_{window}'] = (
                self.data['Discharge(m3/s)'].rolling(window).std()
            )
        # Feature columns
        self.feature_cols = [col for col in self.data.columns if col != 'Discharge(m3/s)']

        for col in self.feature_cols:
            col_min = self.data[col].min()
            col_max = self.data[col].max()
            self.data[col] = (self.data[col] - col_min) / (col_max - col_min)

        # Remove rows with NaN
        self.data = self.data.dropna()
        
        
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        """
        Return (features, target, target_date) for the window starting at idx.
        """
        start_idx = idx
        end_idx = start_idx + self.sequence_length
        
        # Features (shape: [sequence_length, num_features])
        features = self.data[self.feature_cols].iloc[start_idx:end_idx].values
        
        # Target index (one step ahead by prediction_horizon)
        target_idx = end_idx + self.prediction_horizon - 1
        target = self.data['Discharge(m3/s)'].iloc[target_idx]  # this is log1p(discharge)
        
        # Also grab the corresponding date label
        target_date = self.data.index[target_idx]
        
        return (
            torch.FloatTensor(features),
            torch.FloatTensor([target]),
            target_date
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


def train_mann_transformer(model, dataset, num_epochs=100, batch_size=32, 
                          learning_rate=1e-4, device='cpu'):
    """
    Training loop that also prints, for each query example:
      - the date,
      - the actual (raw) discharge,
      - the predicted (raw) discharge,
      along with the batch loss.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # mse_loss = nn.MSELoss()
    huber_loss = nn.HuberLoss(delta=1.0)  # delta is adjustable; 1.0 is standard

    train_losses = []
    memory_write_frequency = 10
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        few_shot_batches = dataset.get_few_shot_batch(num_tasks=batch_size)
        for batch_idx, task in enumerate(few_shot_batches):
            optimizer.zero_grad()
            
            # --- Support set forward pass (for memory writes) ---
            support_x = task['support_x'].to(device)       # [support_size, seq_len, features]
            support_y = task['support_y'].to(device)       # [support_size, 1]
            _, _, support_attention = model(support_x, return_attention=True)
            
            if batch_idx % memory_write_frequency == 0:
                with torch.no_grad():
                    support_repr = torch.mean(support_attention, dim=0)  # [memory_size]
                    for i in range(min(support_repr.size(0), 5)):
                        model.mann_transformer.memory_bank.write(support_repr[i])
            
            # --- Query set forward pass  ---
            query_x = task['query_x'].to(device)    # [query_size, seq_len, features]
            query_y = task['query_y'].to(device)    # [query_size, 1]
            query_dates = task['query_dates']       # list of length query_size
            
            query_pred, query_uncertainty = model(query_x)
            
            # Compute losses
            support_loss = huber_loss(_, _.detach())
             # we don’t need support_pred here
            query_loss = huber_loss(query_pred.squeeze(), query_y.squeeze())
            uncertainty_loss = torch.mean(query_uncertainty)
            total_loss       = support_loss + query_loss + 0.1 * uncertainty_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            # --- Print date + true vs. predicted in raw discharge units ---
            # First convert tensors to 1D Python lists (values on log1p scale)
            true_log1p  = query_y.squeeze().detach().cpu().numpy().tolist()      # each=log1p(true)
            pred_log1p  = query_pred.squeeze().detach().cpu().numpy().tolist()   # each=log1p(pred)
            
            # Invert log1p to get raw discharge: expm1(log1p(val)) = val_raw
            true_raw = [float(np.expm1(v)) for v in true_log1p]
            pred_raw = [float(np.expm1(v)) for v in pred_log1p]
            
            print(f"Epoch {epoch}, Batch {batch_idx} — Loss: {total_loss.item():.6f}")
            for i, dt in enumerate(query_dates):
                # Format date as YYYY-MM-DD
                date_str = dt.strftime("%Y-%m-%d")
                print(f"    {date_str}  |  True: {true_raw[i]:.4f}  →  Pred: {pred_raw[i]:.4f}")
            print("")  # blank line between batches
        
        scheduler.step()
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_losses.append(avg_loss)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}\n")
    
    return train_losses


def prepare_discharge_data(file_path):
    """
    Prepare the discharge data for MANN-Transformer training
    (now dropping any rows where 'Discharge(m3/s)' is NaN)
    """
    # Load data
    data = pd.read_excel(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    
    # Drop rows with missing discharge values
    data = data.dropna(subset=['Discharge(m3/s)'])
    
    # Log-transform for handling extreme values
    data['Discharge(m3/s)'] = np.log1p(data['Discharge(m3/s)'])
    
    return data


# Main execution with error handling
if __name__ == "__main__":
        # Initialize model and dataset
        data = prepare_discharge_data('DOC-20250406-WA0050..xlsx')
        dataset = HydrologicalDataset(data, sequence_length=30, prediction_horizon=1)
        
        # Model configuration
        input_dim = len(dataset.feature_cols)
        model = DischargePredictor(
            input_dim=input_dim,
            d_model=256,
            nhead=8,
            num_layers=6,
            memory_size=100,
            memory_dim=128,
            output_dim=1
        )
        
        # Automatic device detection
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Training with fixed implementation
        train_losses = train_mann_transformer(model, dataset, num_epochs=20, device=device)
        
        print("Training completed successfully!")
        print(f"Final loss: {train_losses[-1]:.6f}")
        
   
