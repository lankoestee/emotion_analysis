import sys
import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import BertTokenizer

from weibodataset import WeiboSentiDataset
from model.parallel import EmotionalAnalysis

def validate(model, val_dataloader, device, epoch=None):
    with torch.no_grad():
        model.eval()
        accs = []
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            outputs = model(input_ids, attention_mask)
            acc = ((outputs > 0.5).int() == labels).float().mean()
            accs.append(acc.item())
        if epoch is not None:
            print(f'epoch {epoch} val acc {sum(accs) / len(accs):.4f}')
        else:
            print(f'test acc {sum(accs) / len(accs):.4f}')
        return sum(accs) / len(accs)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    val_dataset = WeiboSentiDataset('./data/val.csv', tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    model = EmotionalAnalysis()
    model.load_state_dict(torch.load('./model.pth'))
    model.to(device)
    validate(model, val_dataloader, device)