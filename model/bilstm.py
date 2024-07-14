import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import PositionalEncoding

# 定义 Self-Attention 层
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        # x: (batch_size, sequence_length, hidden_dim)
        # mask: (batch_size, sequence_length), padding部分为0,其他部分为1
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 对padding部分进行mask，对其赋值为负无穷

        attention_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        context = torch.bmm(attention_weights, V)
        return context

# 定义 BiLSTM 模型
class BiLSTMwithAttention(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=150,
        output_hidden=128, 
        num_classes=1
    ):
        super(BiLSTMwithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        # self.attention = SelfAttention(hidden_dim * 2)  # 注意力机制应用于BiLSTM的输出
        self.positional_encoding = PositionalEncoding(hidden_dim*2, 0)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, 4, batch_first=True)
        # self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(hidden_dim * 2, output_hidden)
        self.fc2 = nn.Linear(output_hidden, num_classes)

    def forward(self, x, mask=None):
        # x的形状: (batch_size, sequence_length, embedding_dim)
        x, _ = self.lstm(x)
        # lstm_out的形状: (batch_size, sequence_length, 2 * hidden_dim)

        # 应用self-attention机制
        x = self.positional_encoding(x)
        x, _ = self.attention(x, x, x)
        # context的形状: (batch_size, sequence_length, 2 * hidden_dim)

        # 将context传递给全连接层，调整输出维度
        # out = self.fc(context)
        # out的形状: (batch_size, sequence_length, output_dim)
        # x = self.pool(x.permute(0, 2, 1)).squeeze(dim=-1)
        # x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc2(x))
        return x
    
    def output_channels(self):
        return self.hidden_dim * 2