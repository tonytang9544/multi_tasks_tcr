import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler
import random
from pathlib import Path
from paths import DATA_DIR
from sklearn.model_selection import train_test_split

import sceptr
from config import MTDNNConfig
from torch.utils.data import Dataset
from libtcrlm import schema
from libtcrlm.schema import Tcr
from libtcrlm.tokeniser.token_indices import DefaultTokenIndex
from torch.nn import utils
import numpy as np

TASK_NAME_TO_ID = {
    'celltype_binary': 0,
    'binding_binary': 1,
    'celltype_multi' : 2,
    'binding_multi' : 3
}


class TcrDataset(Dataset):
    def __init__(self, data: DataFrame):
        super().__init__()
        self._tcr_series = schema.generate_tcr_series(data)

    def __len__(self) -> int:
        return len(self._tcr_series)

    def __getitem__(self, index: int) -> Tcr:
        return self._tcr_series.iloc[index]

class SingleTaskDataset(Dataset):
    def __init__(self, dataframe, task_name):
        self.dataframe = dataframe
        self.task_name = task_name  # Store the task name
        #self.labels = dataframe['label'].values  # Assuming the label column is named 'label'
        #self.features = dataframe.drop(columns=['label']).values  # All other columns are features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        task_id = TASK_NAME_TO_ID[self.task_name] 
        return {
            'sample' : item,
            'task_id': task_id
        }
    
    def get_task_id(self):
        return TASK_NAME_TO_ID[self.task_name] 
    
class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets  # Dictionary of datasets with task names as keys

    def __len__(self):
        sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        #import pdb; pdb.set_trace()
        task_id, sample_id = idx
        item = self.datasets[task_id%2].dataframe.iloc[sample_id]
        return { 
            'sample' : item,
            'task_id' : task_id
        }

class MTDNNMultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, mix_opt = 0, extra_task_ratio = 0):
        self._datasets = datasets
        self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        train_data_list = []
        random.seed(10)
        np.random.seed(10)
        for dataset in datasets:
            train_data_list.append(
                self._get_shuffled_index_batches(len(dataset), batch_size)
            )
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        #import pdb; pdb.set_trace
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(
            self._train_data_list, self._mix_opt, self._extra_task_ratio
        )
        
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices
    
class MTDNNCollator:
    def __init__(self, tokeniser):
        self.tokeniser = tokeniser

    def multi_task_collate_fn(self, batch):
        task_id = batch[0]['task_id']
        new_batch = []
        for sample in batch:
            assert sample["task_id"] == task_id
            new_batch.append(sample["sample"])
        batch = new_batch
        batch_df = pd.DataFrame(batch)

        batch_meta = {}
        batch_meta['task_id'] = task_id
        batch_meta['labels'] = torch.tensor(batch_df['label'].values, dtype=torch.long)

        tcr_dataset = TcrDataset(batch_df)
        tokenised_data = []
        for tcr in tcr_dataset._tcr_series:
            tokenised_data.append(self.tokeniser.tokenise(tcr))
        padded_batch = utils.rnn.pad_sequence(
            sequences=tokenised_data, batch_first=True, padding_value=DefaultTokenIndex.NULL
        )
        return (batch_meta, padded_batch)

        
class MTDNNDataProcess:
    def __init__(self, config: MTDNNConfig, tokeniser):
        self.config = config
        self.tokeniser = tokeniser 

        self.train_set_dict, self.valid_set_dict, self.test_set_dict = self.load_dataframes()

    def load_dataframes(self):

        celltype_df= pd.read_csv(DATA_DIR/f'preprocessed/{self.config.celltype_task}.csv')

        binding_train_df = pd.read_csv(DATA_DIR/f'preprocessed/{self.config.binding_task}_train_one_v_all.csv')
        binding_valid_df = pd.read_csv(DATA_DIR/f'preprocessed/{self.config.binding_task}_valid_one_v_all.csv')
        binding_test_df = pd.read_csv(DATA_DIR/f'preprocessed/{self.config.binding_task}_test_one_v_all.csv')
        
        # Calculate the proportions for validation and test sets
        validation_ratio = 0.1  
        test_ratio = 0.1        

        # Calculate the sizes for validation and test sets based on the train size
        validation_size = int((self.config.celltype_train_size / 0.8) * validation_ratio)
        test_size = int((self.config.celltype_train_size / 0.8) * test_ratio)

        # Create the train set
        celltype_train_df, remaining_set = train_test_split(celltype_df, train_size=self.config.celltype_train_size, stratify=celltype_df['label'], random_state=10)

        # Now split the remaining data into validation and test sets
        celltype_valid_df, celltype_test_df = train_test_split(remaining_set, train_size=validation_size, test_size=test_size, stratify=remaining_set['label'], random_state=10)
       
        celltype_train_df.to_csv(DATA_DIR/f'preprocessed/benchmarking/{self.config.celltype_task}_train.csv')
        celltype_valid_df.to_csv(DATA_DIR/f'preprocessed/benchmarking/{self.config.celltype_task}_valid.csv')
        celltype_test_df.to_csv(DATA_DIR/f'preprocessed/benchmarking/{self.config.celltype_task}_test.csv')
       
        train_datasets = {
            self.config.celltype_task: celltype_train_df,
            self.config.binding_task: binding_train_df
        }

        valid_datasets = {
            self.config.celltype_task: celltype_valid_df,
            self.config.binding_task: binding_valid_df
        }

        test_datasets = {
            self.config.celltype_task: celltype_test_df,
            self.config.binding_task: binding_test_df
        }
       
        return train_datasets, valid_datasets, test_datasets
    
    def get_trainloader(self):
        train_datasets = []
        for task_name, dataframe in self.train_set_dict.items():
            train_datasets.append(SingleTaskDataset(dataframe, task_name))
        multitask_train_dataset = MultiTaskDataset(train_datasets)
        batch_sampler = MTDNNMultiTaskBatchSampler(train_datasets, batch_size=self.config.batch_size)
        multi_task_collate_fn = MTDNNCollator(self.tokeniser).multi_task_collate_fn
        dataloader = DataLoader(multitask_train_dataset, batch_sampler=batch_sampler, collate_fn=multi_task_collate_fn)

        return dataloader
    
    def get_val_test_loaders_list(self):
        multi_task_collate_fn = MTDNNCollator(self.tokeniser).multi_task_collate_fn

        val_datasets = []
        val_dataloaders = []
        test_datasets = []
        test_dataloaders = []

        for task_name, dataframe in self.valid_set_dict.items():
            val_datasets.append(SingleTaskDataset(dataframe, task_name))
            
        for dataset in val_datasets:
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=multi_task_collate_fn)
            val_dataloaders.append(dataloader)

        for task_name, dataframe in self.test_set_dict.items():
            test_datasets.append(SingleTaskDataset(dataframe, task_name))
        for dataset in test_datasets:
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=multi_task_collate_fn)
            test_dataloaders.append(dataloader)

        return val_dataloaders, test_dataloaders


