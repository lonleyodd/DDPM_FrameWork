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
import time
from .utils import set_save_path,set_logger


class BasePipeline:
    '''
        prepare for train pipeline
    '''
    def __init__(self, *args, **kwargs):
        self.save_dir = set_save_path(args[0]["save_dir"])
        self.logger = set_logger(self.save_dir["save_root"])
        self.writer = SummaryWriter(self.save_dir["tensorboard_path"])


class Pipeline(BasePipeline):
    '''
        this pipeline is used for training target model
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(args[0])
        self.model = kwargs["model"]
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.train_dataset = kwargs["train_dataset"]
        self.test_dataset = kwargs["test_dataset"]
        self.val_dataset = kwargs["val_dataset"]
        
        cfg=args[0]
        self.checkpoint = cfg.checkpoint
        self.epoch = cfg.epoch
        self.batch_size = cfg.batch_size
        self.eval_iters = cfg.eval_iters
        self.save_iters = cfg.save_iters
        self.current_iter = 0
        
        #  distribute training
        train_sampler = None
        if kwargs["Distribution"]:
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

        self.global_iters = self.epoch*len(self.train_dataloader)

    def load_weight(self):
        if self.checkpoint is not None:
            self.logger.info(f"is loading checkpoint from {self.checkpoint}")
            self.model = self.model.load_state_dict(torch.load(self.checkpoint))
        self.model.train()
        self.model = self.model.to(self.device)

    def batch_to_device(self, data):
        for i in range(len(data)):
            data[i].to(self.device)
        return data

    def train(self,epoch):
        for idx, data in enumerate(self.train_dataloader):
            data = self.batch_to_device(data)
            t1 =time.time()
            output = self.model(data[0],data[1])
            t  =time.time()-t1
            loss = output["loss"]

            # if not self.scheduler:
            #     self.optimizer.param_groups["lr"]=self.get_lr(self.global_iters)

            self.logger.info("epoch:{}/{}, iters:{}/{},loss:{:.4f}, time:{:.4f}s, lr:{}".format(
                        epoch,
                        self.epoch,
                        self.current_iter,
                        self.global_iters,
                        loss.item(),
                        t,
                        self.scheduler.optimizer.param_groups[-1]['lr'],
                ))
            
            self.writer.add_scalar("train/loss",self.current_iter,loss.item())

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            # if self.scheduler:
            self.scheduler.step()

            self.current_iter += 1

            if (self.current_iter + 1) % self.eval_iters == 0:
                self.eval()
            if (self.current_iter + 1) % self.save_iters == 0:
                self.save_checkpoint()

    def eval(self):
        self.model.eval()
        for idx,data in enumerate(self.val_dataset):
            data=self.batch_to_device(data)
            output=self.model(data)
            logits=output['logits']


    def start(self):
        # self.load_weight()
        for i in range(self.epoch):
            self.train(i)
            
            self.eval()

