import pandas as pd
from torchinfo import summary

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


from dataset_utils import generate_all_three_cdrs
from sceptrFineTuneModel import SceptrFineTuneModel, cdr_tokenise

CD_label_col = "CD4_or_CD8"

tcr_data_path = "~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz"

tc_df = pd.read_csv(tcr_data_path).dropna().reset_index(drop=True)
print(tc_df.head())


hidden_dim_1 = 128*2
hidden_dim_2 = 64*4

    
model = SceptrFineTuneModel()
summary(model)


# aa_sequences = generate_all_three_cdrs(tc_df)
# print(model(cdr_tokenise(aa_sequences).to("cuda")).shape)

tc_df["label"] = LabelEncoder().fit_transform(tc_df["CD4_or_CD8"])

train, test = train_test_split(tc_df, test_size=0.2, random_state=42)

train, val = train_test_split(train, test_size=0.2, random_state=42)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

num_epochs = 20
batch_size = 1024*4
num_train_batches = int(train.shape[0] / batch_size)
num_val_batches = int(val.shape[0] / batch_size)
num_test_batches = int(test.shape[0] / batch_size)

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for i in range(num_train_batches):
#         # Clear the gradients
#         optimizer.zero_grad()
#         batch = train.iloc[i*batch_size: (i+1)*batch_size]
        
#         # Forward pass
#         outputs = model(cdr_tokenise(batch).to(model.device)).squeeze()
#         loss = criterion(outputs, torch.tensor(batch["label"].to_numpy(), dtype=torch.float32).to(model.device))
        
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#     # scheduler.step()
    
#     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/batch_size:.4f}')

#     model.eval()
#     with torch.no_grad():
#         running_loss = 0.0

#         for i in range(num_val_batches):
#             batch = val.iloc[i*batch_size: (i+1)*batch_size]
            
#             # Forward pass
#             outputs = model(cdr_tokenise(batch).to(model.device)).squeeze()
#             loss = criterion(outputs, torch.tensor(batch["label"].to_numpy(), dtype=torch.float32).to(model.device))
#             running_loss += loss.item()
#         print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {running_loss/batch_size:.4f}')


# print('Training complete')

# torch.save(model, "model.pt")

# model = SceptrFineTuneModel()
model = torch.load("model.pt", weights_only=False)

model.eval()

# Collect all predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(num_test_batches):
        batch = test.iloc[i*batch_size: (i+1)*batch_size]
        
        # Forward pass
        outputs = model(cdr_tokenise(batch).to(model.device)).squeeze()
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(batch["label"].to_numpy())

# Calculate AUC
auc = roc_auc_score(all_labels, all_preds)
print(f'AUC: {auc}')
RocCurveDisplay.from_predictions(all_labels, all_preds)
plt.savefig("AUC plot")
plt.clf()
plt.close()