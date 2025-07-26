import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import array
import matplotlib.pyplot as plt

# --- 0. CONFIGURATION ---
# Fix random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# --- Define file paths and key column names ---
FILE_PATH = './mlp_MANN/new data.xlsx'  # Main dataset with all features
# *** NEW: Path to the raw, unfiltered test data for the target variable ***
RAW_TEST_FILE_PATH = './mlp_MANN/rawtesty.csv' # <--- IMPORTANT: REPLACE IF YOUR FILENAME IS DIFFERENT

DATE_COLUMN = 'Date'
# *** MODIFIED FOR Hirakud with 4 inputs ***
INPUT_FEATURES = ['Mean_areal_rainfall_upper', 'Inflow ', 'Sundargarh', 'Kurubhata']
TARGET_FEATURE = 'Inflow '

# Model and training parameters
N_STEPS_IN, N_STEPS_OUT = 30, 5 # Look back 30 days to predict next 5 days
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001


# --- 1. DATA PREPARATION ---
print("--- 1. Loading and Preparing Data for Hirakud Prediction ---")
try:
    data = pd.read_excel(FILE_PATH)
    print("Successfully loaded data from Excel file.")
except FileNotFoundError:
    print(f"ERROR: Dataset not found at '{FILE_PATH}'.")
    print("Please update the FILE_PATH variable with the correct path to your Excel file.")
    exit()

# Select and rename columns for clarity
data = data[[DATE_COLUMN] + INPUT_FEATURES]
data.rename(columns={
    'Mean_areal_rainfall_upper': 'rainfall',
    'Inflow ': 'hirakud_discharge', # This is our target
    'Sundargarh': 'sundargarh_discharge',
    'Kurubhata': 'kurubhata_discharge'
}, inplace=True)

# Update feature names based on renaming
TARGET_FEATURE_NAME = 'hirakud_discharge'
FEATURES_TO_USE = ['rainfall', 'hirakud_discharge', 'sundargarh_discharge', 'kurubhata_discharge']
data = data[[DATE_COLUMN] + FEATURES_TO_USE]

# Convert 'Date' column to datetime objects and set as index
data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
data.set_index(DATE_COLUMN, inplace=True)

# Impute missing values
print(f"Missing values before imputation: {data.isnull().sum().sum()}")
data.ffill(inplace=True); data.bfill(inplace=True)
print(f"Missing values after imputation: {data.isnull().sum().sum()}")

# --- 2. DATA SPLITTING & PREPROCESSING ---
print("\n--- 2. Splitting, Smoothing, and Scaling Data ---")

# Split data into training and testing sets based on the specified dates
# data_train = data.loc['2000-01-01':'2010-12-31'].copy()
# data_train = data.loc['2002-01-01':'2010-12-31'].copy()
data_train = data.loc['2005-01-01':'2010-12-31'].copy()
data_test = data.loc['2011-01-01':'2014-12-31'].copy()
print(f"Training data shape: {data_train.shape} (from {data_train.index.min()} to {data_train.index.max()})")
print(f"Testing data shape: {data_test.shape} (from {data_test.index.min()} to {data_test.index.max()})")

# *** IMPORTANT: Keep a copy of the raw (un-smoothed) test data for later comparison ***
raw_data_test_target = data_test[[TARGET_FEATURE_NAME]].copy()

# Smoothing the data using a Blackman window
for col in FEATURES_TO_USE:
    train_values = data_train[col].values
    N = 20; window = np.blackman(N)
    smoothed_train = np.convolve(window / window.sum(), train_values, mode='same')
    data_train.loc[:, col] = smoothed_train
    
    test_values = data_test[col].values
    smoothed_test = np.convolve(window / window.sum(), test_values, mode='same')
    data_test.loc[:, col] = smoothed_test
print("Data smoothing complete.")

# Scaling the data to a range of [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)
train_scaled = scaler.transform(data_train)
test_scaled = scaler.transform(data_test)
print("Data scaling complete.")

# --- 3. SEQUENCE GENERATION ---
print("\n--- 3. Creating Input/Output Sequences ---")

target_col_index = data_train.columns.get_loc(TARGET_FEATURE_NAME)
print(f"Target feature '{TARGET_FEATURE_NAME}' is at column index: {target_col_index}")

def split_sequences(sequences, n_steps_in, n_steps_out, target_idx):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences): break
        seq_x = sequences[i:end_ix, :]
        seq_y = sequences[end_ix:out_end_ix, target_idx]
        X.append(seq_x); y.append(seq_y)
    return array(X), array(y)

X, y = split_sequences(train_scaled, N_STEPS_IN, N_STEPS_OUT, target_col_index)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=seed)
print(f"Train shapes: X={train_X.shape}, y={train_y.shape}")
print(f"Validation shapes: X={val_X.shape}, y={val_y.shape}")

train_loader = DataLoader(TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)

# --- 4. MODEL DEFINITION (MLP+MANN) ---
class MLP_MANN_Model(nn.Module):
    def __init__(self, input_dim, n_steps_in, n_steps_out, memory_size=128, memory_dim=64, hidden_dim=256):
        super(MLP_MANN_Model, self).__init__()
        flattened_input_dim = n_steps_in * input_dim
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        nn.init.xavier_uniform_(self.memory)
        self.input_projection = nn.Linear(flattened_input_dim, memory_dim)
        self.dense1 = nn.Linear(flattened_input_dim + memory_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, n_steps_out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        flat_x = x.view(x.size(0), -1)
        projected_input = self.input_projection(flat_x)
        attention_scores = torch.matmul(projected_input, self.memory.t())
        attention_weights = self.softmax(attention_scores)
        memory_output = torch.matmul(attention_weights, self.memory)
        augmented_input = torch.cat([flat_x, memory_output], dim=1)
        h1 = self.relu(self.dense1(augmented_input))
        h2 = self.relu(self.dense2(h1))
        output = self.output_layer(h2)
        return output

# --- 5. TRAINING LOOP ---
print("\n--- 5. Starting Model Training ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device, model_path):
    model.to(device)
    best_val_loss = float('inf')
    train_loss_history, val_loss_history = [], []
    for epoch in range(epochs):
        model.train(); running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for val_X_batch, val_y_batch in val_loader:
                val_X_batch, val_y_batch = val_X_batch.to(device), val_y_batch.to(device)
                val_outputs = model(val_X_batch)
                val_loss += criterion(val_outputs, val_y_batch).item()
        avg_val_loss = val_loss / len(val_loader)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> Validation loss improved. Model saved to '{model_path}'")
    return train_loss_history, val_loss_history

# Instantiate and train the model
input_dim = X.shape[-1] # This will correctly be 4, based on FEATURES_TO_USE
model = MLP_MANN_Model(input_dim=input_dim, n_steps_in=N_STEPS_IN, n_steps_out=N_STEPS_OUT)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
MODEL_SAVE_PATH = "best_mlp_mann_model_Hirakud.pth"

train_loss_hist, val_loss_hist = train_model(model, train_loader, val_loader, EPOCHS, criterion, optimizer, device, MODEL_SAVE_PATH)

# --- 6. POST-TRAINING ANALYSIS ---
print("\n--- 6. Plotting Training and Validation Loss ---")
if train_loss_hist and val_loss_hist:
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_hist, label='Train Loss', color='blue')
    plt.plot(val_loss_hist, label='Validation Loss', color='orange')
    plt.xlabel('Epochs'); plt.ylabel('Loss (MSE)')
    plt.title('MLP+MANN (Hirakud): Training and Validation Loss', fontsize=16)
    plt.legend(); plt.grid(True); plt.show()

# --- 7. FINAL EVALUATION ON UNSEEN TEST SET ---
print("\n--- 7. Evaluating Model on Unseen Test Data (2011-2014) ---")

X_test, y_test_scaled = split_sequences(test_scaled, N_STEPS_IN, N_STEPS_OUT, target_col_index)
print(f"Generated {X_test.shape[0]} test samples from the smoothed data.")

# Load the best model for Hirakud
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(device); model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    test_predictions_scaled = model(X_test_tensor).cpu().numpy()

def inverse_scale_data(scaled_data, original_scaler, n_features, target_idx):
    # Create a dummy array of the same shape as the original data
    dummy_data = np.zeros((scaled_data.size, n_features))
    # Place the scaled data into the target column
    dummy_data[:, target_idx] = scaled_data.flatten()
    # Inverse transform the entire dummy array
    inv_data_full = original_scaler.inverse_transform(dummy_data)
    # Extract only the target column we care about
    inv_data = inv_data_full[:, target_idx]
    # Reshape it back to the original (samples, n_steps_out) shape
    return inv_data.reshape(scaled_data.shape)

n_features = train_scaled.shape[1] # This will be 4
# These are the SMOOTHED actuals and predictions, returned to their original scale
y_test_inv = inverse_scale_data(y_test_scaled, scaler, n_features, target_col_index)
test_predictions_inv = inverse_scale_data(test_predictions_scaled, scaler, n_features, target_col_index)
print("Inverse scaling of smoothed predictions and smoothed true values complete.")

# --- 7.5. APPLY OFFSET BASED ON RAW DATA (NEW SECTION) ---
print("\n--- 7.5. Applying Offset to Align with Raw Data ---")

# Step 1: Load the raw, unfiltered test data from the separate file
try:
    # Assuming the Excel file has no header row, as in the image
    raw_y_test_df = pd.read_csv(RAW_TEST_FILE_PATH, header=0, dtype=float)
    raw_y_test_from_file = raw_y_test_df.values
    print(f"Successfully loaded raw test data from '{RAW_TEST_FILE_PATH}'. Shape: {raw_y_test_from_file.shape}")
except FileNotFoundError:
    print(f"ERROR: Raw test data file '{RAW_TEST_FILE_PATH}' not found. Aborting.")
    exit()
except Exception as e:
    print(f"Error reading raw test data file: {e}")
    exit()

# Step 2: Trim all data to match the 450 data points (90 samples) from the raw file
num_raw_samples = raw_y_test_from_file.shape[0] # Should be 90
print(f"Trimming data to the first {num_raw_samples} samples ({num_raw_samples*N_STEPS_OUT} data points) to match the provided raw data file.")

# The true, raw values to compare against
y_test_raw = raw_y_test_from_file[:num_raw_samples, :]
# The smoothed actual values (from the original script)
y_test_smoothed = y_test_inv[:num_raw_samples, :]
# The model's predictions (based on smoothed data)
y_predict_smoothed = test_predictions_inv[:num_raw_samples, :]

# Step 3: Find the offset
# offset = raw test - smoothed test
offset = y_test_raw - y_test_smoothed
print("Calculated offset between raw and smoothed test data.")

# Step 4: Add the offset to the model's predictions
y_predict_offset_adjusted = y_predict_smoothed + offset
print("Offset applied to model predictions.")


# --- 8. PERFORMANCE METRICS & PLOTTING (MODIFIED) ---
print("\n--- 8. Calculating Final Performance Metrics against RAW Data ---")

# The 'observed' values are now the true raw values.
# The 'predicted' values are the new offset-adjusted predictions.
observed_final = y_test_raw
predicted_final = y_predict_offset_adjusted

test_rmse = np.sqrt(mean_squared_error(observed_final, predicted_final))
print(f"Overall Test RMSE (Adjusted vs Raw): {test_rmse:.4f}")

nse_values, rsr_values, pbias_values, tpe_values = [], [], [], []
for step in range(N_STEPS_OUT):
    # Use the final adjusted values for metrics
    observed = observed_final[:, step]
    predicted = predicted_final[:, step]
    
    if np.sum(observed) == 0 or np.sum((observed - np.mean(observed)) ** 2) == 0:
        nse, rsr, pbias, tpe = -np.inf, np.inf, np.inf, np.inf
    else:
        nse = 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))
        rsr = np.sqrt(np.sum((observed - predicted) ** 2)) / np.sqrt(np.sum((observed - np.mean(observed)) ** 2))
        pbias = (np.sum(predicted - observed) / np.sum(observed)) * 100
        tpe = (np.sum(np.abs(observed - predicted)) / np.sum(observed)) * 100
    nse_values.append(nse); rsr_values.append(rsr)
    pbias_values.append(pbias); tpe_values.append(tpe)

print("\nMetrics for each prediction step (t+1 to t+5):")
print(f"NSE Values    : {np.round(nse_values, 4)}")
print(f"RSR Values    : {np.round(rsr_values, 4)}")
print(f"PBIAS (%)     : {np.round(pbias_values, 4)}")
print(f"TPE (%)       : {np.round(tpe_values, 4)}")

# Plot test set predictions vs actuals
print("\n--- Plotting Final Test Set Results (Adjusted vs. Raw) ---")
prediction_step_to_plot = 0 # Plotting the first day (t+1)
plt.figure(figsize=(18, 8))
# Blue line: The true, unfiltered raw discharge values
plt.plot(observed_final[:, prediction_step_to_plot], color='blue', label='Actual Discharge (Raw, Unfiltered)')
# Red line: The model's prediction after adding the offset
plt.plot(predicted_final[:, prediction_step_to_plot], color='red', label='Predicted Discharge (Offset Adjusted)', alpha=0.8)
# Green dotted line (optional but useful): The model's original "smooth" prediction
plt.plot(y_predict_smoothed[:, prediction_step_to_plot], color='green', linestyle='--', label='Original Predicted Discharge (Smoothed)', alpha=0.6)

plt.title(f'Hirakud Test Set: Final Comparison for Step t+{prediction_step_to_plot + 1}', fontsize=16)
plt.xlabel(f'Time (Sample Index - {num_raw_samples} total samples shown)')
plt.ylabel('Discharge (Original Scale)')
plt.legend(); plt.grid(True);plt.show()