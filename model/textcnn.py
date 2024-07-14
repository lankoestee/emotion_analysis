import torch
from torch import nn
from torch.nn import functional as F

from model.utils import PositionalEncoding

class TextCNN(nn.Module):
    def __init__(
        self, 
        input_embed_dim = 768,
        kernel_size = [3, 5, 7],
        output_channels = 128,
        dropout = 0.4,
        output_hidden = 128,
        num_heads = 4,
        num_classes = 1,
        use_embedding = False,
        tokenizer = None,
        **kwargs
    ):
        super(TextCNN, self).__init__()
        self.mid_channels = output_channels * len(kernel_size)
        self.num_heads = num_heads
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Embedding(tokenizer.vocab_size, input_embed_dim)
        # 卷积层
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=input_embed_dim, out_channels=output_channels, kernel_size=k, padding=(k-1)//2) for k in kernel_size])
        # self.pool = nn.ModuleList([nn.AdaptiveMaxPool1d(1) for _ in kernel_size])
        self.dropout = nn.Dropout(dropout)
        # 位置编码
        self.positional_encoding = PositionalEncoding(self.mid_channels, 0)
        # 多头自注意力
        self.attention = nn.MultiheadAttention(self.mid_channels, num_heads, batch_first=True)
        # 输出层
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(self.mid_channels, output_hidden)
        self.fc2 = nn.Linear(output_hidden, num_classes)

    def forward(self, x, attention_mask):
        seq_len = x.shape[1]
        if self.use_embedding:
            x = x.int()
            x = self.embedding(x)
        # x: [batch_size, seq_len, embedding_dim]
        x = x.permute(0, 2, 1)
        # x: [batch_size, embedding_dim, seq_len]
        x = [F.relu(conv(x)) for conv in self.conv]
        # x = [pool(i).squeeze(dim=-1) for i, pool in zip(x, self.pool)]
        x = torch.cat(x, dim=1)
        # x: [batch_size, mid_channels, seq_len]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        # x: [batch_size, seq_len, mid_channels]
        x = self.positional_encoding(x)
        # attention_mask = attention_mask.bool()
        # attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)
        # attention_mask = attention_mask.unsqueeze(2).expand(-1, -1, seq_len)
        x, _ = self.attention(x, x, x)
        # x = self.pool(x.permute(0, 2, 1)).squeeze(dim=-1)
        # # x: [batch_size, seq_len, mid_channels]
        # x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc2(x))
        return x
    
    def output_channels(self):
        return self.mid_channels

# text_cnn = TextCNN()
# x = torch.randn(16, 200, 768)
# output = text_cnn(x)
