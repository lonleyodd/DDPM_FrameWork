from dataset.text_dataset import PretrainDataset
from model.transformer import Tranformer,CustomLRScheduler
from frame.train import Pipeline
import yaml
import torch
from easydict import EasyDict

def main():
    # build your model
    with open("cfg/llm.yaml",'r') as f:
        cfg=yaml.load(f,Loader=yaml.FullLoader)
    cfg = EasyDict(cfg)
    
    model = Tranformer(cfg.Model)

    opt_cfg   = cfg.Optimizer
    optimizer = model.param_configure(opt_cfg.weight_decay, opt_cfg.lr, (opt_cfg.beta1, opt_cfg.beta2), 'cuda')
    sche_cfg  = cfg.Scheduler
    scheduler = CustomLRScheduler(sche_cfg, optimizer)

    
    # build your dataset
    train_dataset=PretrainDataset(cfg.Dataset)
    test_dataset =None
    val_dataset  =None

    #build your optimizer
    trainer_cfg=cfg.Trainer
    pipeline=Pipeline(
        trainer_cfg,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        Distribution=False
    )

    pipeline.start()

if __name__=="__main__":
    main()