import pandas as pd
import numpy as np
from torchinfo import summary
import pickle

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sceptr


train_config_dict = {
    "lr": 1e-3,
    "num_epoch": 50,
    "classifier_hid_dim": 128,
    "batch_size": 1024*4,
    "dataset_path": "~/Documents/results/data_preprocessing/TABLO/TABLO_full_sceptr_nr_cdr.csv.gz",
    "sceptr_model": "default"
}

sceptr_model_variants = {
    "default": sceptr.variant.default,
    "large": sceptr.variant.large,
    "small": sceptr.variant.small,
    "tiny": sceptr.variant.tiny
}

print("training parameters:")
print(train_config_dict)

label_col = "MAIT_or_NOT"

tcr_data_path = train_config_dict["dataset_path"]

tc_df = pd.read_csv(tcr_data_path).dropna().reset_index(drop=True)#.iloc[:1000]
tc_df[label_col] = tc_df["annotation_L3"] == "MAIT"

print(tc_df.head())
print(tc_df[label_col].unique())
print(tc_df[label_col].value_counts())

tc_df["label"] = LabelEncoder().fit_transform(tc_df[label_col])

train_val, test = train_test_split(tc_df, test_size=0.2, random_state=42)

train, val = train_test_split(train_val, test_size=0.2, random_state=42)

sceptr_model = sceptr_model_variants[train_config_dict["sceptr_model"]]()

# get model encoder output (=input) dimension
sceptr_model_dim = sceptr_model._bert.get_vector_representations_of(
            torch.tensor([[
                [0, 0, 0, 0],
                [1, 2, 3, 2],
                [4, 3, 3, 2]
        ]]).to(sceptr_model._device)).shape[1]

# define classifier model
class TCellClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(TCellClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.tensor(sceptr_model.calc_vector_representations(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))
        return x
    
model = TCellClassifier(input_dim=sceptr_model_dim)
print(summary(model))


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=train_config_dict["lr"])
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

num_epochs = train_config_dict["num_epoch"]
batch_size = train_config_dict["batch_size"]
num_train_batches = int(train.shape[0] / batch_size)
num_val_batches = int(val.shape[0] / batch_size)
num_test_batches = int(test.shape[0] / batch_size)

best_val_loss = np.inf

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i in range(num_train_batches):
        # Clear the gradients
        optimizer.zero_grad()
        batch = train.iloc[i*batch_size: (i+1)*batch_size]
        
        # Forward pass
        outputs = model(batch).to(sceptr_model._device).squeeze()
        loss = criterion(outputs, torch.tensor(batch["label"].to_numpy(), dtype=torch.float32).to(sceptr_model._device))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    # scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/num_train_batches:.4f}')

    model.eval()
    with torch.no_grad():
        running_loss = 0.0

        for i in range(num_val_batches):
            batch = val.iloc[i*batch_size: (i+1)*batch_size]
            
            # Forward pass
            outputs = model(batch).to(sceptr_model._device).squeeze()
            loss = criterion(outputs, torch.tensor(batch["label"].to_numpy(), dtype=torch.float32).to(sceptr_model._device))
            running_loss += loss.item()
        curr_val_loss = running_loss/num_val_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {curr_val_loss:.4f}')
        if curr_val_loss < best_val_loss:
            torch.save(model, "model.pt")
            best_val_loss = curr_val_loss

print('Training complete')

model = torch.load("model.pt", weights_only=False).eval()

# Collect all predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(num_test_batches):
        batch = test.iloc[i*batch_size: (i+1)*batch_size]
        outputs = model(batch).to(sceptr_model._device).squeeze()
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(batch["label"].to_numpy())


with open("true_pred_dict.pkl", "wb") as f:
    pickle.dump(
        {
            "y_true": all_labels,
            "y_pred": all_preds
        },
        f
    )


print(f'ROC_AUC: {roc_auc_score(all_labels, all_preds)}')
RocCurveDisplay.from_predictions(all_labels, all_preds)
plt.savefig("ROC_AUC plot")
plt.clf()
plt.close()
print(f"Average precision score: {average_precision_score(all_labels, all_preds)}")
PrecisionRecallDisplay.from_predictions(all_labels, all_preds)
plt.savefig("Precision Recall curve plot")
plt.clf()
plt.close()