import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split, default_collate
from triplet_loss import *
from Siamese import SiameseModel, QuadrupletModel
import lightning.pytorch as pl
#from lightning.pytorch.loggers import wandb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from utils import TextDataset, AdvDataset, load_feature_extractor
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from itertools import cycle
import argparse

class ParallelCollector:
    #Add random sampled out-of-distribution samples with corresponding labels
    def __init__(self, ood_non_vul_loader, ood_vul_loader):
        self.ood_non_vul_loader = ood_non_vul_loader
        self.ood_vul_loader = ood_vul_loader
    
    def __call__(self, data):
        x_train, y_train, x_train_tr, _ = default_collate(data)
        x_train_ood = torch.concat(
            [next(iter(self.ood_non_vul_loader))[0] if y == 1 else next(iter(self.ood_vul_loader))[0] for y in y_train], axis=0)
        return (x_train, x_train_tr, x_train_ood), y_train

def train(model_type, feature_extractor, compacted_train_loader, compacted_val_loader):
    model_dict = {"sia" : SiameseModel, "quad" : QuadrupletModel}
    model = model_dict[model_type](feature_extractor)
    #logger = wandb.WandbLogger(project='inputReflector', name=model_type, log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=True, mode='min')
    #trainer = pl.Trainer(callbacks=[early_stopping,checkpoint_callback], logger=logger, max_epochs=200, devices="auto", accelerator="auto")
    trainer = pl.Trainer(callbacks=[early_stopping,checkpoint_callback], max_epochs=200, devices="auto", accelerator="auto")
    trainer.fit(model, train_dataloaders=compacted_train_loader, val_dataloaders=compacted_val_loader)
    return model


def main():
    parser = argparse.ArgumentParser(description='Input Reflector')
    parser.add_argument('--ood_non_vul_file', type=str, default='reveal_non_vul.jsonl', help='OOD non-vulnerable file')
    parser.add_argument('--ood_vul_file', type=str, default='reveal_vul.jsonl', help='OOD vulnerable file')
    parser.add_argument('--pretrained_model', type=str, default='model.bin', help='pretrained codebert model')
    parser.add_argument('--train_file', type=str, default='adv_examples.jsonl', help='train file')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--model_type', choices=['sia', 'quad'])
    args = parser.parse_args()
    
    print('Loading Model')
    tokenizer, feature_extractor = load_feature_extractor(args.pretrained_model)
    block_size = tokenizer.max_len_single_sentence

    print('Loading Datasets')
    ood_non_vul_set = TextDataset(tokenizer, block_size, args.ood_non_vul_file)
    ood_non_vul_loader = DataLoader(ood_non_vul_set, batch_size=1, sampler=RandomSampler(ood_non_vul_set, replacement=True))
    ood_vul_set = TextDataset(tokenizer, block_size, args.ood_vul_file)
    ood_vul_loader = DataLoader(ood_vul_set, batch_size=1, sampler=RandomSampler(ood_vul_set, replacement=True))

    #Split Adv Samples into train_set and validate set
    adv_full_set = AdvDataset(tokenizer, block_size, args.train_file)
    lengths = [int(p * len(adv_full_set)) for p in [0.9,0.1]]
    lengths[-1] = len(adv_full_set) - sum(lengths[:-1])
    
    adv_train_set, adv_val_set = random_split(
        adv_full_set,
        lengths=lengths
    )
    
    collate_fn = ParallelCollector(ood_non_vul_loader, ood_vul_loader)
    adv_train_loader = DataLoader(adv_train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn = collate_fn, num_workers=4)
    adv_val_loader = DataLoader(adv_val_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn = collate_fn, num_workers=4)
    
    print(f'Ready to Train {args.model_type}')
    model = train(args.model_type, feature_extractor, adv_train_loader, adv_val_loader)

if __name__ == '__main__':
    main()





