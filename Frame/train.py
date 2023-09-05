'''
author: Ryze
'''

import os
import logging
import platform
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.distributed as dist

from utils import *


class BasePipeline:
    '''
        prepare for train pipeline
    '''

    def __init__(self, *args, **kwargs):
        self.save_dir = set_save_path(kwargs['save_dir'])
        self.logger = set_logger(self.save_dir["save_root"])
        self.writer = SummaryWriter(self.save_dir["tensorboard_path"])


class Pipeline(BasePipeline):
    '''
        this pipeline is used for training target model
    '''
    def __init__(self, *args, **kwargs):
        super(Pipeline).__init__(*args, **kwargs)
        self.model = kwargs["model"]
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.train_dataset = kwargs["train_dataset"]
        self.val_dataset = kwargs["val_dataset"]
        self.checkpoint = kwargs["checkpoint"]
        self.epoch = kwargs["epoch"]
        self.batch_size = kwargs["batch_size"]

        self.eval_step = kwargs["eval_step"]

        self.global_step = 0

        #  distribute training
        train_sampler = None
        if kwargs["Distribute_training"]:
            local_rank = int(kwargs["local_rank"])
            torch.cuda.set_device(local_rank)
            if platform.system() == "linux":
                dist.init_process_group(backend='nccl')
            elif platform.system() == "windows":
                dist.init_process_group(backend='gloo')
            else:
                raise TypeError(f"this framework is not support {platform.system()}")

            train_sampler = DistributedSampler(kwargs["train_dataset"])
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=4
        )
        self.val_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def load_weight(self):
        if self.checkpoint is not None:
            self.logger.info(f"is loading checkpoint from {self.checkpoint}")
            self.model = self.model.load_state_dict(torch.load(self.checkpoint))
        self.model.train()
        self.model = self.model.to(self.device)

    def batch_to_device(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        return data


    def train(self):

        for idx, data in enumerate(self.train_dataset):
            data = self.batch_to_device(data)

            output = self.model(data)

            loss = output["loss"]
            self.writer.add_scalar("train/loss",self.global_step,loss.item(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            if (self.global_step + 1) % self.eval_step == 0:
                self.eval()
        self.eval()


    def eval(self):
        self.model.eval()
        for idx,data in enumerate(self.val_dataset):
            data=self.batch_to_device(data)
            output=self.model(data)
            logits=output['logits']


    def pipeline(self):
        self.load_weight()

        for i in range(self.epoch):
            self.train()

            self.eval()

