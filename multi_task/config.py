import torch
import torch.nn as nn
from paths import *
import pandas as pd

class MTDNNConfig:
    def __init__(
        self,
        seed = 10,
        # model architecture
        pretrained_model_name ='sceptr',
        hidden_size=64,
        # tasks
        celltype_task = "celltype_binary",
        binding_task = "binding_binary",
        celltype_train_size=500,
        single_task = '',
        # training 
        learning_rate=5e-5,
        epochs=3,
        batch_size=8,
        cuda=torch.cuda.is_available(),
        cuda_device=0,
        grad_accumulation_steps=1,
        optimizer="adamax",
        warmup_proportion=0.1,
        warmup_schedule="warmup_linear",
        adam_eps=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=50,
        save_steps=1000,
        # scheduler 
        have_lr_scheduler=True,
        multi_step_lr="10,20,30",
        lr_gamma=0.5,
        num_train_step = -1,
        model_checkpoint_dir=PROJECT_ROOT/"mtdnn_checkpoints",
    ):
        self.seed = seed
        
        # Model architecture parameters
        self.pretrained_model_name = pretrained_model_name
        self.hidden_size = hidden_size

        # Task specific parameters 
        self.celltype_task = celltype_task
        self.binding_task = binding_task 
        self.celltype_train_size = celltype_train_size
        self.single_task = single_task 

        # Training parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.cuda = cuda
        self.cuda_device = cuda_device
        self.grad_accumulation_steps = grad_accumulation_steps
        self.optimizer = optimizer
        self.warmup_proportion = warmup_proportion
        self.warmup_schedule = warmup_schedule
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        # Scheduler parameters
        self.have_lr_scheduler = have_lr_scheduler
        self.multi_step_lr = multi_step_lr
        self.lr_gamma = lr_gamma
        self.num_train_step = num_train_step 

        # Logging and saving
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.model_checkpoint_dir = model_checkpoint_dir
