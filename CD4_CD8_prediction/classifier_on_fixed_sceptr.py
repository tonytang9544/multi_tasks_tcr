import pandas as pd
import numpy as np 
import torch

import sceptr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle



CD_label_col = "CD4_or_CD8"

tcr_data_path = "~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr.csv.gz"


tc_df = pd.read_csv(tcr_data_path)

subjects = tc_df['donor'].unique()

subject_embeddings = {}
for subject in subjects:
    # Filter DataFrame for the current subject
    df_subject = tc_df[tc_df['donor'] == subject]
    
    # Compute embeddings
    embeddings = sceptr.calc_vector_representations(df_subject)
    
    # Store embeddings and labels in the dictionary
    subject_embeddings[subject] = {
        "embeddings": embeddings,
        "labels": df_subject[CD_label_col].to_numpy()
    }

# Flatten the dictionary into a list of rows
data = []
for subject, content in subject_embeddings.items():
    embeddings = content['embeddings']
    labels = content['labels']
    for embedding, label in zip(embeddings, labels):
        data.append({'subject': subject, CD_label_col: label, 'embedding': embedding})


embed_df = pd.DataFrame(data)

label_encoder = LabelEncoder()
embed_df['labels_encoded'] = label_encoder.fit_transform(embed_df[CD_label_col])
embeddings = torch.tensor(np.stack(embed_df['embedding'].values), dtype=torch.float32)
labels = torch.tensor(embed_df['labels_encoded'].values, dtype=torch.float32)

# Assuming embeddings is your tensor of shape (num_samples, embedding_dim)
normalized_embeddings = F.normalize(embeddings, p=2, dim=1)


# Assuming X is your feature matrix and y is your labels
X_train, X_test, y_train, y_test = train_test_split(normalized_embeddings, labels, test_size=0.2, random_state=42, stratify=labels)

# Convert tensors to numpy arrays before using SMOTE
X_train_np = X_train.numpy()
y_train_np = y_train.numpy()

# Try SMOTE
#smote = SMOTE()
#X_train_resampled_np, y_train_resampled_np = smote.fit_resample(X_train_2_np, y_train_2_np)
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled_np, y_train_resampled_np = undersampler.fit_resample(X_train_np, y_train_np)
# print(X_train_resampled_np.shape)

# Convert the numpy arrays back to tensors
X_train_resampled = torch.tensor(X_train_resampled_np, dtype=torch.float32)
y_train_resampled = torch.tensor(y_train_resampled_np, dtype=torch.float32)

# Create the TensorDataset
train_dataset = TensorDataset(X_train_resampled, y_train_resampled)
# test_dataset = TensorDataset(X_test_2, y_test_2)
test_dataset = TensorDataset(X_test, y_test)


batch_s = 1024*4
hidden_dim_1 = 128*2
hidden_dim_2 = 64*4


train_loader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=False)

class TCellClassifier(nn.Module):
    def __init__(self, input_dim):
        super(TCellClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        return x

    
input_dim = X_train.shape[1]
model = TCellClassifier(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

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
    # scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

print('Training complete')

torch.save(model, "model.pt")

model.eval()

# Collect all predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        #print(outputs)
        #print(outputs.squeeze().size())
        #print(all_preds)
        all_preds.extend(outputs.squeeze().numpy())
        all_labels.extend(labels.numpy())

# Calculate AUC
auc = roc_auc_score(all_labels, all_preds)
print(f'AUC: {auc}')
RocCurveDisplay.from_predictions(all_labels, all_preds)
plt.savefig("AUC plot")
plt.clf()
plt.close()

with open("prediction_label_dictionary.pkl", "wb") as f:
    pickle.save(
        {
            "all_preds": all_preds,
            "all_labels": all_labels
        },
        f
    )