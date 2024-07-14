import torch
import torch.nn as nn

from model.utils import PositionalEncoding

class TransformerCNN(nn.Module):
    def __init__(
        self,
        input_size=768,
        output_size=512,
        num_layers=2,
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        cnn_out_channels=512,
        cnn_kernel_size=3,
    ):
        super(TransformerCNN, self).__init__()

        self.input_size = input_size # 输入特征维度
        self.hidden_size = hidden_size # 隐藏层维度
        self.num_layers = num_layers # Transformer的层数

        # CNN layer
        self.cnn = nn.Conv1d(input_size, cnn_out_channels, cnn_kernel_size, padding=cnn_kernel_size//2) # 使用一维卷积
        
        # Embedding layer to project CNN output to hidden size
        self.embedding = nn.Linear(cnn_out_channels, hidden_size) # CNN输出的维度映射到Transformer的隐藏层维度
        
        self.positional_encoding = PositionalEncoding(hidden_size, 0.1) # 位置编码
        
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout) # Transformer的encoder层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers) 
        

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 这里是为了将输入的维度变换为(batch_size, input_size, seq_length)
        x = self.cnn(x) # 传入CNN层
        x = x.permute(0, 2, 1)  # 将维度变换为(batch_size, seq_length, cnn_out_channels)
        x = self.embedding(x) # 传入embedding层
        x = self.positional_encoding(x) # 传入位置编码层
        x = x.permute(1, 0, 2)  # 适应Transformer的输入维度
        output = self.transformer_encoder(x) # 传入Transformer
        output = output.permute(1, 0, 2)  # 将维度变换为(batch_size, seq_length, hidden_size)
        return output
    
    def output_channels(self):
        return self.hidden_size
