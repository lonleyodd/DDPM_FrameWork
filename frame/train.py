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
        self.val_dataset = kwargs["val_dataset"]
        
        cfg=args[0]
        self.ddp=cfg.ddp
        self.checkpoint_path = cfg.checkpoint_path
        self.epoch = cfg.epoch
        
        self.batch_size = cfg.batch_size
        self.eval_iters = cfg.eval_iters
        self.save_iters = cfg.save_iters
        self.current_iter = 0

        self.current_epoch = 0
        
        #  distribute training
        train_sampler = None
        if self.ddp:
            self.logger.info("start ddp progress...")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size=int(os.environ["WORLD_SIZE"])
            self.rank= int(os.environ["RANK"])
            
            device=f"cuda:{self.local_rank}"
            torch.cuda.set_device(device)
            if platform.system() == "Linux":
                dist.init_process_group(backend='nccl',world_size=self.world_size)
            elif platform.system() == "Windows":
                dist.init_process_group(backend='gloo',world_size=self.world_size)
            else:
                raise TypeError(f"this framework is not support {platform.system()}")

            train_sampler = DistributedSampler(kwargs["train_dataset"])
            val_sampler = DistributedSampler(kwargs["train_dataset"])

            self.logger.info(f"world size: {self.world_size}, progress: {self.local_rank},rank: {self.rank} is ready for training")
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
            dataset=self.val_dataset,
            sampler=val_sampler,
            batch_size=self.batch_size,
            num_workers=4
        )
        self.global_iters = self.epoch*len(self.train_dataloader)

    def load_weight(self):
        if self.checkpoint is not None:
            self.logger.info(f"is loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            self.model = self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer = self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_iter=checkpoint["iter"]
            self.current_epoch=checkpoint["epoch"]
        self.model = self.model.to(self.device)

    def batch_to_device(self, data):
        for i in range(len(data)):
            data[i].to(self.device)
        return data

    def save_checkpoint(self,):
        checkpoint_name="epoch_{}_iters_{}_checkpoint.pt"
        checkpoint={
            "epoch": self.current_epoch,
            "iter": self.current_iter,
            "model_state_dict": self.model.state_dict,
            "optimizer_state_dict":self.optimizer.state_dict,
        }
        torch.save(checkpoint,os.path.join(self.save_dir["checkpoint_path"],checkpoint_name))

    def train(self,):
        self.model.train()
        for idx, data in enumerate(self.train_dataloader):
            data = self.batch_to_device(data)
            t1 =time.time()
            output = self.model(data[0],data[1])
            t  =time.time()-t1
            loss = output["loss"]

            # if not self.scheduler:
            #     self.optimizer.param_groups["lr"]=self.get_lr(self.global_iters)
            if self.ddp:
                loss=dist.all_reduce(loss,op=dist.ReduceOp.SUM)
                print(loss)
                loss=loss/self.world_size

            self.logger.info("epoch:{}/{}, iters:{}/{},loss:{:.4f}, time:{:.4f}s, lr:{:.6f}".format(
                        self.current_epoch,
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
                perplexity=self.eval()
                self.writer.add_scalar("train/perplexity",self.current_iter,perplexity)
                self.logger.info("epoch:{}, iters:{}, perplexity:{}".format(epoch,self.current_iter,perplexity))

            if (self.current_iter + 1) % self.save_iters == 0:
                self.save_checkpoint()


    def eval(self):
        self.model.eval()
        losses=0
        for idx,data in enumerate(self.val_dataset):
            data_cuda=self.batch_to_device(data)
            with torch.no_grad():
                output=self.model(data_cuda)
            loss=output["loss"]
            losses+=loss.float()
        losses=losses/(idx+1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        self.model.train()

        return perplexity

    def start(self):
        # self.load_weight()
        for i in range(self.epoch):
            
            self.train()
            
            self.eval()

            

