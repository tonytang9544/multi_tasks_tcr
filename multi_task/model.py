import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import numpy as np
import json
from pytorch_pretrained_bert import BertAdam as Adam
import random
import os
from sklearn.metrics import roc_auc_score

import sceptr
from config import MTDNNConfig
from dataset import MTDNNDataProcess, TASK_NAME_TO_ID
from get_num_model_parameters import get_model_dimensionality
import hugging_face_lms
from optim import Adamax, RAdam, AdamWithScheduling

ID_TO_TASK_NAME = {v: k for k, v in TASK_NAME_TO_ID.items()}

class MultiTaskTCRModel(nn.Module):
    def __init__(self, config: MTDNNConfig):
        super(MultiTaskTCRModel, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if self.config.cuda else "cpu")

        # Define shared encoder params 
        if self.config.pretrained_model_name == 'sceptr':
            model = sceptr.variant.default()
            self.config.hidden_size = get_model_dimensionality(model)
            self.shared_encoder = model._bert
            self.tokeniser = model._tokeniser
        elif self.config.shared_model == 'tcr_bert':
            model = hugging_face_lms.TcrBert()
            self.config.hidden_size = get_model_dimensionality(model)
            self.shared_encoder = model._model
            self.shared_encoder = model._get_tokeniser()
        elif self.config.shared_model == 'esm2':
            model = hugging_face_lms.Esm2()
            self.config.hidden_size = get_model_dimensionality(model)
            self.shared_encoder = model._model
            self.shared_encoder = model._get_tokeniser()
        elif self.config.shared_model == 'protbert':
            model = hugging_face_lms.ProtBert()
            self.config.hidden_size = get_model_dimensionality(model)
            self.shared_encoder = model._model
            self.shared_encoder = model._get_tokeniser()
        else:
            raise RuntimeError(f"Unsupported pretrained model: {self.config.pretrained_model_name}")
        
        
        # Define celltype classifier 
        if self.config.celltype_task == "celltype_binary":
            self.celltype_classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, 2),  # Fully connected layer
                nn.ReLU(),
                nn.Dropout(0.1)
        )
        elif self.config.celltype_task == "celltype_multilabel":
            self.celltype_classifier = nn.Linear(self.config.hidden_size, 10)
        else:
            raise RuntimeError(f"Unsupported cell type task: {self.config.celltype_task}")

        # Define binding classifier 
        if self.config.binding_task == "binding_binary":
            self.binding_classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, 2),  # Fully connected layer
                nn.ReLU(),
                nn.Dropout(0.1)
        )
        if self.config.binding_task == "binding_multi":
            self.binding_classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, 6),  # Fully connected layer
                nn.ReLU(),
                nn.Dropout(0.1)
        )

        if self.config.single_task == '':
            self.celltype_classifier = self.celltype_classifier
            self.binding_classifier = self.binding_classifier
        elif self.config.single_task == 'celltype':
            self.binding_classifier = None
        elif self.config.single_task == 'binding':
            self.celltype_classifier = None 
        else:
            raise RuntimeError(f"Unsupported single task type: {self.config.single_task}")


    def forward_celltype(self, cell_type_inputs): 
        outputs = self.shared_encoder.get_vector_representations_of(cell_type_inputs)
        celltype_logits = self.celltype_classifier(outputs)
        return celltype_logits

    def forward_binding(self, binding_inputs):
        outputs = self.shared_encoder.get_vector_representations_of(binding_inputs)
        binding_logits = self.binding_classifier(outputs)
        return binding_logits

    def _get_param_groups(self):
        no_decay = ["bias", "gamma", "beta", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters


class MTDNNTCR:
    def __init__(self, config: MTDNNConfig):
        self.config = config
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.network = MultiTaskTCRModel(config)
        if self.config.cuda:
            self.network.cuda(device=self.config.cuda_device)
        
        self.tokeniser = self.network.tokeniser
        data_process = MTDNNDataProcess(self.config, self.tokeniser)
        self.train_loader = data_process.get_trainloader()
        self.val_loader_list, self.test_loader_list = data_process.get_val_test_loaders_list()
       
        self.optimizer_parameters = self.network._get_param_groups()
        #self.network.shared_encoder.set_fine_tuning_mode(True)
        self._setup_optim(self.optimizer_parameters)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer.zero_grad()
 
        #import pdb; pdb.set_trace()

    def _setup_optim(self, optimizer_parameters, state_dict: dict = None, num_train_step: int = -1):
        # Setup optimizer parameters
        if self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                optimizer_parameters,
                self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamax":
            self.optimizer = Adamax(
                optimizer_parameters,
                self.config.learning_rate,
                warmup=self.config.warmup_proportion,
                t_total=num_train_step,
                max_grad_norm=self.config.max_grad_norm,
                schedule=self.config.warmup_schedule,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "radam":
            self.optimizer = RAdam(
                optimizer_parameters,
                self.config.learning_rate,
                warmup=self.config.warmup_proportion,
                t_total=num_train_step,
                max_grad_norm=self.config.max_grad_norm,
                schedule=self.config.warmup_schedule,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adam":
            self.optimizer = Adam(
                optimizer_parameters,
                lr=self.config.learning_rate,
                warmup=self.config.warmup_proportion,
                t_total=num_train_step,
                max_grad_norm=self.config.max_grad_norm,
                schedule=self.config.warmup_schedule,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adam_with_scheduling":
            self.optimizer = AdamWithScheduling(
                optimizer_parameters,
                d_model=self.config.hidden_size,
                n_warmup_steps=int((self.config.epochs * len(self.train_loader)) * self.config.warmup_proportion),
                lr=self.config.learning_rate,
                lr_multiplier=1,
                decay=False
            )
        else:
            raise RuntimeError(f"Unsupported optimizer: {self.config.optimizer}")

        # Clear scheduler for certain optimizer choices
        if self.config.optimizer in ["adam", "adamax", "radam", "adam_with_scheduling"]:
            if self.config.have_lr_scheduler:
                self.config.have_lr_scheduler = False

        if state_dict and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.config.have_lr_scheduler:
            if self.config.warmup_schedule == "rop":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, mode="max", factor=self.config.lr_gamma, patience=3
                )
            elif self.config.warmup_schedule == "exp":
                self.scheduler = ExponentialLR(
                    self.optimizer, gamma=self.config.lr_gamma or 0.95
                )
            else:
                milestones = [
                    int(step)
                    for step in (self.config.multi_step_lr or "10,20,30").split(",")
                ]
                self.scheduler = MultiStepLR(
                    self.optimizer, milestones=milestones, gamma=self.config.lr_gamma
                )
        else:
            self.scheduler = None

    def train(self):        
        device = torch.device("cuda" if self.config.cuda else "cpu")
        self.network.to(device)

        training_metrics = [] #store all metrics for each epoch
        best_val_accuracies = {}
        best_val_losses = {}
        best_epochs = {}
        best_avg_val_accuracy = 0.0  # To track the best average validation accuracy
        best_avg_val_loss = 10000000000
        best_avg_epoch = 0  # To track the epoch with the best average validation accuracy
        

        #celltype_loss_weight = 0.2
        #binding_loss_weight = 0.8

        for val_loader in self.val_loader_list:
                dataset_task_name = val_loader.dataset.task_name
                best_val_accuracies[dataset_task_name] = 0.0
                best_val_losses[dataset_task_name] = 10000000000
                best_epochs[dataset_task_name] = 0

        # global_step = 0
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1}/{self.config.epochs}")

            # Train phase
            self.network.train()
            total_train_loss = 0.0
            all_train_predictions = []
            all_train_labels = []
            num_train_batches = 0 
            
            for (batch_meta, batch_data) in self.train_loader:
                batch_data = batch_data.to(device)
                labels = batch_meta['labels'].to(device)

                task_id = batch_meta['task_id']
                task_name = ID_TO_TASK_NAME[task_id]
                
                if self.config.single_task == 'celltype' and task_id not in [0,2]:
                    continue  # Skip non-celltype batch
                if self.config.single_task == 'binding' and task_id not in [1, 3]:
                    continue # Skip non-binding batch
                
                if task_id in [0,2]: #celltype task according to task name to ID dictionary
                    logits = self.network.forward_celltype(batch_data)
                    assert task_name == self.config.celltype_task
                    #loss_weight = celltype_loss_weight
                else:
                    logits = self.network.forward_binding(batch_data)
                    assert task_name == self.config.binding_task
                    #loss_weight = binding_loss_weight
                
                self.optimizer.zero_grad()
               
                #import pdb; pdb.set_trace()
                # Compute loss
                loss = self.criterion(logits, labels) 

                # Compute gradient and update model
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                num_train_batches += 1
              
                train_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                all_train_predictions.extend(train_predictions)
                all_train_labels.extend(labels.cpu().numpy())

            avg_train_loss = total_train_loss / num_train_batches
            train_accuracy = (np.array(all_train_predictions) == np.array(all_train_labels)).mean()

            # Validation phase
            self.network.eval()

            dataset_val_metrics = {}
            total_val_accuracy = 0.0
            total_val_loss = 0.0

            if self.config.single_task == '':
                num_datasets = len(self.val_loader_list)
            else:
                num_datasets = 1

            with torch.no_grad():
                for val_loader in self.val_loader_list:
                    dataset_task_name = val_loader.dataset.task_name
                    dataset_task_id = val_loader.dataset.get_task_id()

                    if self.config.single_task == 'celltype' and dataset_task_id not in [0, 2]:
                        continue  # Skip non-celltype task
                    if self.config.single_task == 'binding' and dataset_task_id not in [1, 3]:
                        continue  # Skip non-binding task

                    total_task_val_loss = 0.0
                    all_val_predictions = []
                    all_val_labels = []
                    num_val_batches = 0

                    for (batch_meta, batch_data) in val_loader:
                        batch_data = batch_data.to(device)
                        labels = batch_meta['labels'].to(device)

                        task_id = batch_meta['task_id']
                        assert task_id == dataset_task_id
                
                        if task_id in [0,2]: #celltype task according to task name to ID dictionary
                            logits = self.network.forward_celltype(batch_data)
                    
                        else:
                            logits = self.network.forward_binding(batch_data)
                    
                        # Compute loss
                        val_loss = self.criterion(logits, labels)
                        total_task_val_loss += val_loss.item()
                        num_val_batches += 1

                        val_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                        all_val_predictions.extend(val_predictions)
                        all_val_labels.extend(labels.cpu().numpy())

                    avg_task_val_loss = total_task_val_loss / num_val_batches 
                    val_accuracy = (np.array(all_val_predictions) == np.array(all_val_labels)).mean()

                    # Store validation metrics for this dataset
                    dataset_val_metrics[dataset_task_name] = {
                        'val_loss': avg_task_val_loss,
                        'val_accuracy': val_accuracy
                    }

                    total_val_accuracy += val_accuracy
                    total_val_loss += avg_task_val_loss
                    # Save the best model for each dataset
                    if avg_task_val_loss < best_val_losses[dataset_task_name]:
                        best_val_losses[dataset_task_name] = avg_task_val_loss
                        best_epochs[dataset_task_name] = epoch + 1
                        best_model_save_path = f"{self.config.model_checkpoint_dir}/{self.config.celltype_task}_{self.config.binding_task}/{dataset_task_name}_best_checkpoint.pt"
                        torch.save(self.network.state_dict(), best_model_save_path)
                        print(f"Best model for {dataset_task_name} saved at epoch {epoch + 1} with loss {total_task_val_loss:.4f}")

                # Save the best model based on average val accuracy across tasks 
                avg_val_accuracy = total_val_accuracy / num_datasets
                avg_val_loss = total_val_loss / num_datasets
                
                if avg_val_loss < best_avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    best_avg_epoch = epoch + 1
                    avg_model_save_path = f"{self.config.model_checkpoint_dir}/{self.config.celltype_task}_{self.config.binding_task}/best_avg_checkpoint.pt"
                    torch.save(self.network.state_dict(), avg_model_save_path)
                    print(f"Best model based on average validation accuracy saved at epoch {epoch + 1} with average loss {avg_val_loss:.4f}")     

                # Print dataset-wise validation results
                for dataset_name, metrics in dataset_val_metrics.items():
                    print(f"Validation results for {dataset_name} - Loss: {metrics['val_loss']:.4f}, Accuracy: {metrics['val_accuracy']:.4f}")

            # Save metrics for this epoch
            training_metrics.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'validation_metrics': dataset_val_metrics,
                'avg_val_accuracy': avg_val_accuracy,
                'avg_val_loss': avg_val_loss
            })

            print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Average Validation Accuracy: {avg_val_accuracy:.4f}")   

        # Save the last model
        last_model_save_path = f"{self.config.model_checkpoint_dir}/{self.config.celltype_task}_{self.config.binding_task}/last_checkpoint.pt"
        torch.save(self.network.state_dict(), last_model_save_path)
        print(f"Last model checkpoint saved at {last_model_save_path}")

        # Save overall best validation accuracies and corresponding epochs
        overall_best_metrics = {
            'best_avg_val_accuracy': best_avg_val_accuracy,
            'best_avg_epoch': best_avg_epoch,
            'best_val_accuracies': best_val_accuracies,
            'best_epochs': best_epochs
        }

        # Save metrics to a file
        metrics_save_path = f"{self.config.model_checkpoint_dir}/{self.config.celltype_task}_{self.config.binding_task}/training_validation_metrics.json"
        with open(metrics_save_path, 'w') as metrics_file:
            json.dump({'training_metrics': training_metrics, 'overall_best_metrics': overall_best_metrics}, metrics_file, indent=4)
        print(f"Training and validation metrics saved at {metrics_save_path}")        



                # if global_step % self.config.logging_steps == 0:
                #    print(f"Step {global_step}: {task_name.capitalize()} Loss: {loss.item()}")

                # if global_step % self.config.save_steps == 0:
                #     model_save_path = f"{self.config.model_checkpoint_dir}/model_step_{global_step}.pt"
                #     torch.save(self.network.state_dict(), model_save_path)
                #     print(f"Model saved at {model_save_path}")

                # global_step += 1


    def test(self):
        device = torch.device("cuda" if self.config.cuda else "cpu")
        self.network.to(device)

        test_accuracies = {}

        dataset_test_metrics = {}

        best_model_save_path = f"{self.config.model_checkpoint_dir}/{self.config.celltype_task}_{self.config.binding_task}/best_avg_checkpoint.pt"
        
        # Load the original checkpoint
        checkpoint = torch.load(best_model_save_path)

        # Initialize a new state_dict for the shared encoder
        shared_encoder_state_dict = {}

        # Loop through the original state_dict and filter out the shared encoder parameters
        for key, value in checkpoint.items():
            if key.startswith('shared_encoder.'):
                # Remove 'shared_encoder.' from the key
                new_key = key[len('shared_encoder.'):]
                shared_encoder_state_dict[new_key] = value

        # Save the new checkpoint with only the shared encoder parameters
        new_checkpoint = shared_encoder_state_dict
        torch.save(new_checkpoint, f'{self.config.model_checkpoint_dir}/{self.config.celltype_task}_{self.config.binding_task}/shared_encoder_checkpoint_single.pt')

        # Load the best model checkpoint for this task
        if os.path.exists(best_model_save_path):
            print(f"Loading best overall model from {best_model_save_path}")
            self.network.load_state_dict(torch.load(best_model_save_path, map_location=device))
        else:
            raise RuntimeError(f"Best model checkpoint not found")
        
        self.network.eval()

        with torch.no_grad():
            for test_loader in self.test_loader_list:
                dataset_task_name = test_loader.dataset.task_name
                dataset_task_id = test_loader.dataset.get_task_id()
                test_accuracies[dataset_task_name] = 0.0

                if self.config.single_task == 'celltype' and dataset_task_id not in [0, 2]:
                    continue  # Skip non-celltype task
                if self.config.single_task == 'binding' and dataset_task_id not in [1, 3]:
                    continue  # Skip non-binding task

                total_test_loss = 0.0
                all_test_predictions = []
                all_test_labels = []
                all_test_probs = []  # Collect probabilities for AUROC
                num_test_batches = 0

                for (batch_meta, batch_data) in test_loader:
                    batch_data = batch_data.to(device)
                    labels = batch_meta['labels'].to(device)

                    task_id = batch_meta['task_id']
                    assert task_id == dataset_task_id
            
                    if task_id in [0,2]: #celltype task according to task name to ID dictionary
                        logits = self.network.forward_celltype(batch_data)
                
                    else:
                        logits = self.network.forward_binding(batch_data)
                
                    # Compute loss
                    test_loss = self.criterion(logits, labels)
                    total_test_loss += test_loss.item()
                    num_test_batches += 1

                    # Convert logits to probabilities
                    test_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Take probability for class 1
                    all_test_probs.extend(test_probs)

                    test_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                    all_test_predictions.extend(test_predictions)
                    all_test_labels.extend(labels.cpu().numpy())

                avg_test_loss = total_test_loss / num_test_batches 
                test_accuracy = (np.array(all_test_predictions) == np.array(all_test_labels)).mean()
           
                # Compute AUROC
                if len(np.unique(all_test_labels)) > 1:  # Avoid issues if only one class is present
                    test_auroc = roc_auc_score(all_test_labels, all_test_probs)
                else:
                    test_auroc = float('nan')  # Handle case where only one label is present

                # Store validation metrics for this dataset
                dataset_test_metrics[dataset_task_name] = {
                    'test_loss': avg_test_loss,
                    'test_accuracy': test_accuracy,
                    'test_auroc': test_auroc
                }

                print("Final test metrics for all datasets:", dataset_test_metrics)




