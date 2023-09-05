from .dataset.text_dataset import SquenceDataset
from model.transformer import Tranformer
from Frame.train import Pipeline


def main():
    # build your model
    model=Tranformer()

    # build your dataset
    train_dataset=SquenceDataset()
    test_dataset=SquenceDataset()


    pipeline=Pipeline(model,train_dataset,test_dataset)

    pipeline.train()
