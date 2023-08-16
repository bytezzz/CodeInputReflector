import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from model import Model
import torch.nn.functional as F


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


def convert_examples_to_features(code, tokenizer, block_size, label):
    # source
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)


class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js['func'], tokenizer, block_size, js['target']))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

class AdvDataset(Dataset):
    def __init__(self, tokenizer, block_size, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append((convert_examples_to_features(js['ori_func'], tokenizer, block_size, js['ori_label']), convert_examples_to_features(js['adv_func'], tokenizer, block_size, js['adv_label'])))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i][0].input_ids), torch.tensor(self.examples[i][0].label), torch.tensor(self.examples[i][1].input_ids), torch.tensor(self.examples[i][1].label)
    

def load_feature_extractor(path):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    config = RobertaConfig.from_pretrained("microsoft/codebert-base")
    config.num_labels=1
    feature_extractor = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base",config = config)
    feature_extractor = Model(feature_extractor, tokenizer)
    feature_extractor.load_state_dict(torch.load(path))
    return tokenizer, feature_extractor.encoder

from torch.utils.data import SequentialSampler
def get_results(model, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=4,pin_memory=False)

        model.eval()
        probs=[] 
        labels=[]
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")       
            with torch.no_grad():
                outputs = model(inputs,attention_mask=inputs.ne(1))[0]
                prob = F.sigmoid(outputs)
                probs.append(prob.cpu())
                labels.append(batch[1])
                
        probs=torch.cat(probs,0)
        labels=torch.cat(labels,0)

        pred_labels = (probs > 0.5).to(torch.long).squeeze(dim=1)

        return pred_labels, labels