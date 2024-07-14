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

def train_one_epoch(model, optimizer, dataloader, device, epoch, loss_fn):
    model.train()
    optimizer.zero_grad()
    loss_list = []
    acc_list = []
    dataloader = tqdm(dataloader, file=sys.stdout)
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        labels = labels.unsqueeze(1).float()
        attention_mask = attention_mask.bool()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = ((outputs > 0.5).int() == labels).float().mean()
        loss_list.append(loss.item())
        acc_list.append(acc.item())
        dataloader.set_description(f'epoch {epoch} loss {np.average(loss_list):.4f} acc {np.average(acc_list):.4f}')
    return np.average(loss_list), np.average(acc_list)

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

# 检查是否存在run文件夹，不存在则创建
if not os.path.exists('runs'):
    os.makedirs('runs')
# 检查文件夹内的序号
run_num = len(os.listdir('runs'))
os.makedirs(f'runs/{run_num}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

train_csv = "dataset/train.csv"
val_csv = "dataset/val.csv"
test_csv = "dataset/test.csv"
dataset = WeiboSentiDataset(train_csv, tokenizer)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
val_dataset = WeiboSentiDataset(val_csv, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_dataset = WeiboSentiDataset(test_csv, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = EmotionalAnalysis().to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
best_acc = 0

for epoch in range(30):
    loss_train, acc_train = train_one_epoch(model, optimizer, dataloader, device, epoch, loss_fn)
    acc_val = validate(model, val_dataloader, device, epoch)
    # 追加记录到csv文件
    with open(f'runs/{run_num}/hist.csv', 'a') as f:
        f.write(f'{epoch},{loss_train},{acc_train},{acc_val}\n')
    if acc_val > best_acc:
        best_acc = acc_val
        model.save(f"runs/{run_num}/best.pth")
    model.save(f'runs/{run_num}/last.pth')