import __future__

import numpy as np
import warnings

from sympy import use
import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_model import PerceiverResampler, PerceiverResamplerGlobal, PerceiverResamplerGlobalPosition
from dataset import SOS_TOKEN, EOS_TOKEN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random

class VideoEncoder(nn.Module):
    def __init__(self, input_size=512, window_size=15, framerate=2, pool="PerceiverResamplerGlobalPosition", dropout=0.1, proj_size=768,
                 use_global=True, use_position=True, use_homogenization=True):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,hidden_size)
        """

        super(VideoEncoder, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.framerate = framerate
        self.pool = pool
        self.proj_size = proj_size
        
        if not self.input_size == proj_size:   
            self.feature_extractor = nn.Linear(self.input_size, proj_size)
            input_size = proj_size
            self.input_size = proj_size

        if self.pool == 'PerceiverResamplerGlobalPosition':
            self.hidden_size = input_size
            self.pool_layer = PerceiverResamplerGlobalPosition(dim=input_size, dim_inner=self.hidden_size, 
                                                               use_global=use_global, use_position=use_position, use_homogenization=use_homogenization)

        #self.drop = nn.Dropout(p=0.4)

    def _video_features_extract(self, video_features):
        for i in range(len(video_features)):
            video_features[i] = self.feature_extractor(video_features[i])
        return video_features


    def forward(self, video_features, inputs, positions):
        # input_shape: (batch,frames,dim_features)
        BS, FR, IC = inputs.shape
        if not IC == self.proj_size:
            # inputs = inputs.reshape(BS*FR, IC)
            inputs = self.feature_extractor(inputs)
            video_features = self._video_features_extract(video_features)
            # inputs = inputs.reshape(BS, FR, -1)

        # Temporal pooling operation
        if self.pool == 'PerceiverResamplerGlobalPosition':
            inputs_pooled = self.pool_layer(inputs, video_features, positions)
        return inputs_pooled

def contrastive_loss(text_outputs, vision_outputs, margin=1.0):
    """
    Calculate the contrastive loss for text and vision outputs.
    
    Parameters:
    text_outputs (torch.Tensor): A tensor of shape (batch_size, 1, embedding_dim).
    vision_outputs (torch.Tensor): A tensor of shape (batch_size, 1, embedding_dim).
    margin (float): Margin for the contrastive loss. Default is 1.0.
    
    Returns:
    torch.Tensor: The contrastive loss.
    """
    # Remove the singleton dimension
    a = text_outputs  # shape: (batch_size, embedding_dim)
    b = vision_outputs  # shape: (batch_size, embedding_dim)
    
    distances = torch.cdist(a, b, p=2)
    
    # Create labels: 1 if same index, 0 otherwise
    labels = torch.eye(a.size(0), device=a.device)
    
    # Contrastive loss computation
    positive_loss = labels * distances
    negative_loss = (1 - labels) * F.relu(margin - distances)
    
    loss = positive_loss + negative_loss
    
    # Average the loss over the batch
    loss = loss.sum() / (2 * a.size(0))
    
    return loss


class Video2Classifcation(nn.Module):
    def __init__(self, num_classes, weights=None, input_size=512, window_size=15, framerate=2, pool="QFormer", weights_encoder=None, freeze_encoder=False, proj_size=768):
        super(Video2Classifcation, self).__init__()
        self.encoder = VideoEncoder(input_size, window_size, framerate, pool, proj_size=proj_size, 
                                    use_global=True, use_position=False, use_homogenization=False)
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.num_classes = num_classes
        self.fc = nn.Linear(self.encoder.hidden_size, num_classes)
        self.pool = pool
        

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
    
    def forward(self, video_features, features, positions):
        features = self.encoder(video_features, features, positions)

        if features.dim() == 3:
            features = features.mean(dim=1)
        
        output = self.fc(features)
        return output

class Video2Spot(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, window_size=15, framerate=2, pool="QFormer", weights_encoder=None, freeze_encoder=False, proj_size=768):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(Video2Spot, self).__init__()
        self.encoder = VideoEncoder(input_size, window_size, framerate, pool, proj_size=proj_size, 
                                    use_global=True, use_position=False, use_homogenization=False)
        self.norm = nn.LayerNorm(self.encoder.hidden_size)
        self.head = nn.Sequential(nn.Linear(self.encoder.hidden_size, 32), nn.ReLU(), nn.Linear(32, num_classes+1))
        #self.drop = nn.Dropout(p=0.5)
        self.sigm = nn.Sigmoid()
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

    def forward(self, video_features, inputs, positions):
        # input_shape: (batch,frames,dim_features)
        inputs_pooled = self.encoder(video_features, inputs, positions) # B x 8 x D

        # avg pool in sequence dimension
        if inputs_pooled.dim() == 3:
            inputs_pooled = inputs_pooled.mean(dim=1)
        inputs_pooled = self.norm(inputs_pooled)
        
        # Extra FC layer and squashing
        output = self.head(inputs_pooled)

        return output
    
if __name__ == '__main__':
    model = VideoEncoder(input_size=1024, window_size=30, framerate=1, pool="PerceiverResampler", dropout=0.0, proj_size=768)
    video_features = [torch.rand(1000, 1024)]*2 + [torch.rand(500, 1024)]*2
    inputs = torch.rand(4, 30, 1024)
    out = model(video_features, inputs)
    print(out.shape)