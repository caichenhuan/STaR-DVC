import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import VideoEncoder
import random
from gpt import GPT, GPTConfig
import tiktoken


class Video2CaptionGPT(nn.Module):
    """
    Video2CaptionGPT模型，用于将视频特征和2D位置信息编码为字幕token序列
    """
    def __init__(self, vocab_size, weights=None, input_size=512, window_size=15, framerate=2, pool="NetVLAD", 
                 embed_size=512, hidden_size=512, teacher_forcing_ratio=1, num_layers=2, max_seq_length=300, 
                 weights_encoder=None, freeze_encoder=False, top_k=5,
                 gpt_path="gpt2", gpt_type="gpt2"):
        super(Video2CaptionGPT, self).__init__()
        self.decoder = GPT.from_pretrained(gpt_type, dict(dropout=0.0), path=gpt_path)
        self.encoder = VideoEncoder(input_size, window_size, framerate, pool, proj_size=self.decoder.config.n_embd, 
                                    use_global=True, use_position=True, use_homogenization=True)
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.proj = nn.Linear(self.decoder.config.n_embd, self.decoder.config.n_embd)
        self.top_k = top_k

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))
            
    def load_encoder(self, weights_encoder=None, freeze_encoder=False):
        if(weights_encoder is not None):
            print("=> loading encoder '{}'".format(weights_encoder))
            checkpoint = torch.load(weights_encoder, map_location=torch.device('cpu'))
            self.load_state_dict({k :v for k, v in checkpoint['state_dict'].items() if "encoder." in k}, strict=False)
            print("=> loaded checencoderkpoint '{}' (epoch {})"
                  .format(weights_encoder, checkpoint['epoch']))
            
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
    
    def forward(self, video_features, features, positions, captions, lengths):
        features = self.encoder(video_features, features, positions) # B, T, D
        features = self.proj(features)
        decoder_output, _ = self.decoder(captions, lengths=lengths, features=features)
        return decoder_output
    
    def get_training_parameters(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
        return list(self.encoder.parameters()) + list(self.proj.parameters())
    
    def sample(self, video_features, features, positions, max_seq_length=100):

        features = self.encoder([video_features], features.unsqueeze(0), [positions])
        features = self.proj(features)
        start = ":"
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        start_ids = encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=features.device)[None, ...])
        text = self.decoder.generate(x, max_seq_length, top_k=self.top_k, features=features, eos_token=enc.eot_token)
        text = text[:, 1:]
        return text
    
if __name__ == "__main__":
    model = Video2CaptionGPT(vocab_size=50257, input_size=512, window_size=15, framerate=2, pool="NetVLAD", 
                             embed_size=512, hidden_size=512, teacher_forcing_ratio=1, num_layers=2, max_seq_length=300, 
                             weights_encoder=None, freeze_encoder=False, top_k=5,
                             gpt_path="gpt2", gpt_type="gpt2")
    print(model)