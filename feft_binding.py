from pathlib import Path
import pandas as pd
import sceptr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

tcr_data_path = Path("../preprocessed/") 


binding_binary_train = pd.read_csv(tcr_data_path/"binding_binary_train_one_v_all.csv")
binding_binary_val = pd.read_csv(tcr_data_path/"binding_binary_valid_one_v_all.csv")
binding_binary_test = pd.read_csv(tcr_data_path/"binding_binary_test_one_v_all.csv")


train_embeddings = sceptr.calc_vector_representations(binding_binary_train)
val_embeddings = sceptr.calc_vector_representations(binding_binary_val)
test_embeddings = sceptr.calc_vector_representations(binding_binary_test)

scaler = StandardScaler()
normalized_train_embeddings = scaler.fit_transform(train_embeddings)
normalized_val_embeddings = scaler.fit_transform(val_embeddings)
normalized_test_embeddings = scaler.fit_transform(test_embeddings)

# convert the embeddings and labels to tensors
X_train = torch.tensor(normalized_train_embeddings, dtype=torch.float32)
y_train = torch.tensor(binding_binary_train['label'].values, dtype=torch.float32)
X_val = torch.tensor(normalized_val_embeddings, dtype=torch.float32)
y_val = torch.tensor(binding_binary_val['label'].values, dtype=torch.float32)
X_test = torch.tensor(normalized_test_embeddings, dtype=torch.float32)
y_test = torch.tensor(binding_binary_test['label'].values, dtype=torch.float32)


train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
vak_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class BindingClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BindingClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    
input_dim = X_train.shape[1]
model = BindingClassifier(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Clear the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

print('Training complete')

model.eval()

# Collect all predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        all_preds.extend(outputs.squeeze().numpy())
        all_labels.extend(labels.numpy())

# Calculate AUC
auc = roc_auc_score(all_labels, all_preds)
print(f'AUC: {auc}')


### --- Hyperparam optimisation ---

# Define the hyperparameter grid
param_grid = {
    'learning_rate': [0.001, 0.0001],
    'dropout_rate': [0.1, 0.5],
    'num_epochs': [10, 20],
    'batch_size': [32, 64],
}

# Use ParameterGrid to generate all combinations of hyperparameters
param_combinations = list(ParameterGrid(param_grid))

# Track best model and hyperparameters
best_val_accuracy = 0.0
best_params = None

# Loop through all hyperparameter combinations
for params in param_combinations:
    print(f"Training with parameters: {params}")
    
    # Prepare the data loader with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    # Model, criterion, and optimizer with the current hyperparameters
    input_dim = X_train.shape[1]
    model = BindingClassifier(input_dim, dropout_rate=params['dropout_rate'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Training loop
    num_epochs = params['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            predicted = (outputs >= 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.4f}')

    # Track the best model and hyperparameters
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_params = params

print(f'Best Hyperparameters: {best_params}, Best Validation Accuracy: {best_val_accuracy:.4f}')