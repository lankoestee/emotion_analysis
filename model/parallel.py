import os

import torch
from torch import nn
from torch.nn import functional as F

from transformers import BertTokenizer, BertModel

from model.textcnn import TextCNN
from model.bilstm import BiLSTMwithAttention
from model.transformercnn import TransformerCNN
from model.utils import PositionalEncoding

class EmotionalAnalysis(nn.Module):
    def __init__(self, output_hidden=256, num_classes=1):
        super(EmotionalAnalysis, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        self.bert = BertModel.from_pretrained('./bert-base-chinese')
        # 冻结BERT模型的参数
        for param in self.bert.parameters():
            param.requires_grad = False
        # self.textcnn = TextCNN()
        # self.bilstm = BiLSTMwithAttention()
        self.transformercnn = TransformerCNN()
        self.hidden_size = self.transformercnn.output_channels()
        self.positional_encoding = PositionalEncoding(self.hidden_size, 0)
        self.attention = nn.MultiheadAttention(self.hidden_size, 4, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(self.hidden_size, output_hidden)
        self.fc2 = nn.Linear(output_hidden, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        input = outputs.last_hidden_state
        textcnn_output = self.textcnn(input, attention_mask)
        bilstm_output = self.bilstm(input, attention_mask)
        transformercnn_output = self.transformercnn(input)
        
        # 将TextCNN和BiLSTM以及TransformerCNN的输出拼接
        outputs = torch.cat([textcnn_output, bilstm_output, transformercnn_output], dim=2)
        outputs = self.positional_encoding(outputs)
        outputs, _ = self.attention(outputs, outputs, outputs)
        outputs = self.pool(outputs.permute(0, 2, 1)).squeeze(dim=-1)
        outputs = F.relu(self.fc1(outputs))
        outputs = F.sigmoid(self.fc2(outputs))

        return outputs

    def save(self, path):
        # 不保存BERT模型的参数
        torch.save(self.state_dict(), path)
    
    def active_params(self):
        # 返回可训练的参数量
        return sum(p.numel() for p in self.parameters() if p.requires_grad)