from torch.utils.data import Dataset

import numpy as np
import torch
import os


class BaseDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass


class PretrainDataset(Dataset):
    def __init__(self, cfg,train=False):
        super().__init__()
        if train:
            self.data_path=cfg.train_path
        else:
            self.data_path=cfg.val_path
        self.max_length=cfg.max_length
        

        data_list = []
        with open(self.data_path,'r') as f:
            nbytes = f.seek(0,2)
            flen = f.tell() // np.dtype('uint16').itemsize
        self.data = np.memmap(self.data_path,dtype=np.dtype('uint16'),shape=(flen//self.max_length,self.max_length))
        self.data_length=self.data.shape[0]

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        sample = self.data[item]
        tokens = torch.from_numpy(np.array(sample[:-1]).astype(np.int64))
        targets = torch.from_numpy(np.array(sample[1:]).astype(np.int64))

        # data={"tokens":tokens,"targets":targets}

        return tokens,targets

   


class SFTDataset(Dataset):
    def __init__(self,
                max_length=256,
                prompt_length=128,
                answer_length=128,
                ):
        super().__init__()
        pass        

    def __getitem__(self, item):
        pass


if __name__ == "__main__":
    data = {
        "medical": "F:/data/nlp_dataset/medical/pretrain/train_encyclopedia.json",
        "wikipedia": "F:/data/nlp_dataset/medical/pretrain/train_encyclopedia.json"
    }
    # pretrain_dataset = PretrainDataset(data,)
    with open("E:\code\Train_FrameWork\cache\wikipedia-cn-20230720-filtered.bin", 'r') as f:
        nbytes = f.seek(0, 2)
        flen = f.tell() // np.dtype('uint16').itemsize
    data = np.memmap(r"E:\code\Train_FrameWork\cache\wikipedia-cn-20230720-filtered.bin", dtype=np.dtype('uint16'), shape=(flen // 256, 256))
    print(data)