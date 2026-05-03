
import os
from re import T
import time
import json
import test
import torch
import random
import logging
import tiktoken
import numpy as np
from tqdm import tqdm
from torchtext.vocab import vocab
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from torch.utils.data import default_collate
from torch.utils.data.sampler import Sampler
from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import getMetaDataTask

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    captions = [t[-3] for t in batch]
    idx = [t[-5:-3] for t in batch]
    cls_labels = torch.tensor([t[-2] for t in batch])
    ## padd
    tokens = [([SOS_TOKEN] + t[-6] + [EOS_TOKEN]) if t[-6] else [PAD_TOKEN, PAD_TOKEN] for t in batch]
    tokens = [torch.Tensor(t).long() for t in tokens ]
    ## get sequence lengths
    lengths = torch.tensor([ len(t) for t in tokens ])
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    ## compute mask
    mask = (tokens != PAD_TOKEN)
    ## load video features
    video_features = []
    for t in batch:
        video_features.append(torch.from_numpy(np.squeeze(np.load(t[0], allow_pickle=True).item()['features'])))
        # video_features.append(torch.from_numpy(np.load(t[0], allow_pickle=True)))
    # video_features = torch.stack(video_features)
    ## positions
    positions = [t[-1] for t in batch]
    return default_collate([t[1:-6] for t in batch ]) + [tokens], lengths, mask, captions, idx, cls_labels, video_features, positions

def collate_fn_gpt(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    captions = [t[-3] for t in batch]
    idx = [t[-5:-3] for t in batch]
    cls_labels = torch.tensor([t[-2] for t in batch])
    ## padd
    tokens = [(encode(":") + t[-6] + [enc.eot_token]) if t[-6] else [14841, 14841] for t in batch]
    tokens = [torch.Tensor(t).long() for t in tokens ]
    ## get sequence lengths
    lengths = torch.tensor([len(t) for t in tokens])
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=14841)
    ## compute mask
    mask = (tokens != 14841)
    ## load video features
    video_features = []
    for t in batch:
        video_features.append(torch.from_numpy(np.squeeze(np.load(t[0], allow_pickle=True).item()['features'])))
        # video_features.append(torch.from_numpy(np.load(t[0], allow_pickle=True)))
    ## positions
    positions = [t[-1] for t in batch]
    return default_collate([t[1:-6] for t in batch]) + [tokens], lengths, mask, captions, idx, cls_labels, video_features, positions

def collate_fn_clip(batch):
    # game_ID, feat_half1, feat_half2, label_half1, label_half2, video_feature1, video_feature2, position1, position2
    game_ID = [t[0] for t in batch]
    feat_half1 = torch.stack([t[1] for t in batch], dim=0)
    feat_half2 = torch.stack([t[2] for t in batch], dim=0)

    label_half1 = torch.from_numpy(np.stack([t[3] for t in batch], axis=0))
    label_half2 = torch.from_numpy(np.stack([t[4] for t in batch], axis=0))

    video_feature1 = torch.from_numpy(np.stack([t[5] for t in batch], axis=0))
    video_feature2 = torch.from_numpy(np.stack([t[6] for t in batch], axis=0))

    position1 = [t[7] for t in batch]
    position2 = [t[8] for t in batch]
    return game_ID, feat_half1, feat_half2, label_half1, label_half2, video_feature1, video_feature2, position1, position2

def collate_fn_pred(batch):
    video_features = [t[0] for t in batch]
    vfeats = torch.tensor(np.array([t[1] for t in batch]))
    idx = np.array([t[2] for t in batch])
    caption_id = np.array([t[3] for t in batch])
    positions = [t[4] for t in batch]
    return video_features, vfeats, idx, caption_id, positions


class SoccerNetVideoProcessor(object):
    """video_fn is a tuple of (video_id, half, frame)."""

    def __init__(self, clip_length):
        self.clip_length = clip_length

    def __call__(self, video_fn, feats):
        video_id, half, frame = video_fn
        video_feature = feats[video_id][half]
        # make sure that the clip lenght is right
        start = min(frame, video_feature.shape[0] - self.clip_length)
        video_feature = video_feature[start : start + self.clip_length]

        return video_feature
    
class SoccerNetPositionProcessor(object):
    """video_fn is a tuple of (video_id, half, frame)."""

    def __init__(self, clip_length):
        self.clip_length = clip_length

    def __call__(self, video_fn, positions):
        video_id, half, frame = video_fn
        position = positions[video_id][half]
        # make sure that the clip lenght is right
        start = min(frame,len(position) - self.clip_length)
        position = list(position.values())[start: start+self.clip_length]
        return position

class TikTokenTextProcessor(object):
    def __init__(self, corpus, min_freq=5):
        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        self.decode = lambda l: enc.decode(l)
        self.vocab = [True] * enc.n_vocab

    def __call__(self, text):
        return self.encode(text)

    def detokenize(self, tokens):
        return self.decode(tokens[0].tolist())
        

class SoccerNetTextProcessor(object):
    """
    A generic Text processor
    tokenize a string of text on-the-fly.
    """

    def __init__(self, corpus, min_freq=5):
        import spacy
        spacy_token = spacy.load("en_core_web_sm").tokenizer
        # Add special case rule
        spacy_token.add_special_case("[PLAYER]", [{"ORTH": "[PLAYER]"}])
        spacy_token.add_special_case("[COACH]", [{"ORTH": "[COACH]"}])
        spacy_token.add_special_case("[TEAM]", [{"ORTH": "[TEAM]"}])
        spacy_token.add_special_case("([TEAM])", [{"ORTH": "([TEAM])"}])
        spacy_token.add_special_case("[REFEREE]", [{"ORTH": "[REFEREE]"}])
        self.tokenizer = lambda s: [c.text for c in spacy_token(s)]
        self.min_freq = min_freq
        self.build_vocab(corpus)
    
    def build_vocab(self, corpus):
        counter = Counter([token for c in corpus for token in self.tokenizer(c)])
        voc = vocab(counter, min_freq=self.min_freq, specials=["[PAD]", "[SOS]", "[EOS]", "[UNK]", "[MASK]", "[CLS]"])
        voc.set_default_index(voc['[UNK]'])
        self.vocab = voc
    
    def __call__(self, text):
        return self.vocab(self.tokenizer(text))
    
    def detokenize(self, tokens):
        return " ".join(self.vocab.lookup_tokens(tokens))

class SoccerNetDataset(Dataset):
    """
    This class is used to download and pre-compute clips and captions from the SoccerNet dataset for captining training phase.
    """
    def __init__(self, path, pos_path,features="baidu_soccer_embeddings.npy", split=["train"], version=2, framerate=1, spot_window_size=15, caption_window_size=15, insert_empty_caption=False):
        """
        :param path: 数据集路径
        :param pos_path: 位置数据集路径
        :param features: 特征类型
        :param split: 数据集分割
        :param version: 数据集版本
        :param framerate: 帧率
        :param spot_window_size: 事件定位窗口大小
        :param caption_window_size: 字幕生成窗口大小
        :param insert_empty_caption: 是否插入空字幕
        :return: None
        """
        self.path = path
        self.pos_path = pos_path
        self.split = split
        split = [s for s in split if s!= "challenge"]
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.spot_window_size_frame = spot_window_size*framerate
        self.caption_window_size_frame = caption_window_size*framerate
        self.version = version
        self.use_insert_empty_caption = insert_empty_caption
        self.labels, self.num_classes, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.class_labels = [k for k in self.dict_event.keys()]

        self.data = list()
        self.game_feats = list()
        self.game_positions = list()
        self.video_game_feats_path = list()

        l_pad = self.caption_window_size_frame//2 + self.caption_window_size_frame%2
        r_pad = self.caption_window_size_frame//2 
        looper = self.listGames

        for game_id, game in enumerate(tqdm(looper)):
            # Load features
            feat_half1_path = os.path.join(self.path, game, "1_" + self.features)
            feat_half2_path = os.path.join(self.path, game, "2_" + self.features)
            self.video_game_feats_path.append((feat_half1_path, feat_half2_path))

            if self.features == "224p_5fps.npy":
                feat_half1 = np.squeeze(np.load(feat_half1_path, allow_pickle=True).item()['features'])
                feat_half2 = np.squeeze(np.load(feat_half2_path, allow_pickle=True).item()['features'])
            else:
                feat_half1 = np.load(feat_half1_path)
                feat_half2 = np.load(feat_half2_path)

            feat_half1 = np.pad(feat_half1.reshape(-1, feat_half1.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            feat_half2 = np.pad(feat_half2.reshape(-1, feat_half2.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            
            self.game_feats.append((feat_half1, feat_half2)) 

            # Load 2d positions
            pos_half1_path = os.path.join(self.pos_path, game, "1_sn_position.json")
            pos_half2_path = os.path.join(self.pos_path, game, "2_sn_position.json")
            pos_half1 = json.load(open(pos_half1_path))
            pos_half2 = json.load(open(pos_half2_path))
            self.game_positions.append((pos_half1, pos_half2))

            # Load labels
            last_half = 0
            last_frame = [0, 0]
            caption_id = 0
            labels = json.load(open(os.path.join(self.path, game, self.labels)))
            # sort labels by frame
            labels["annotations"] = sorted(
                labels["annotations"], 
                key=lambda x: (int(x['gameTime'][0]), self._cal_frame_from_time(x['gameTime'], framerate)))
            for idx, annotation in enumerate(labels["annotations"]):
                time = annotation["gameTime"]
                event = annotation["label"]
                half = int(time[0])
                if event not in self.dict_event or half > 2: continue

                # insert empty caption if there is a gap at the end of the game
                frame = self._cal_frame_from_time(time, framerate)
                if self.use_insert_empty_caption:
                    if last_half != half:
                        if last_half == 1:
                            caption_id = self._insert_empty_caption(game_id, last_half, caption_id, last_frame, feat_half1.shape[0])
                        elif last_half == 2:
                            caption_id = self._insert_empty_caption(game_id, last_half, caption_id, last_frame, feat_half2.shape[0])
                        last_half = half

                # TODO: check if there is overlap between two captions
                # if frame - last_frame[half-1] < self.spot_window_size_frame:
                #     # print("overlap: ", last_frame[half-1], frame, frame - last_frame[half-1])
                #     continue

                if half == 1 and frame > feat_half1.shape[0] - self.caption_window_size_frame: continue
                if half == 2 and frame > feat_half2.shape[0] - self.caption_window_size_frame: continue
                if self.use_insert_empty_caption:
                    caption_id = self._insert_empty_caption(game_id, half, caption_id, last_frame, frame)
                # print('class: ', half, caption_id, frame)
                self.data.append(((game_id, half-1, frame) , (caption_id, annotation['anonymized']), self.class_labels.index(event)+1))
                caption_id += 1
        
        #launch a VideoProcessor that will create a clip around a caption
        self.video_processor = SoccerNetVideoProcessor(self.caption_window_size_frame)
        #launch a PositionProcessor that will create a clip
        self.position_processor = SoccerNetPositionProcessor(self.caption_window_size_frame)
        #launch a TextProcessor that will tokenize a caption
        self.text_processor = TikTokenTextProcessor(self.getCorpus(split=["train"]))
        self.vocab_size = len(self.text_processor.vocab)

    def _insert_empty_caption(self, game_id, half, caption_id, last_frame, frame):
        inserted_num = ((frame - last_frame[half-1]) // self.caption_window_size_frame) - 1
        if inserted_num <= 0:
            last_frame[half-1] = frame
            return caption_id
        left_pad = ((frame - last_frame[half-1]) - inserted_num*self.caption_window_size_frame) // 2
        start = last_frame[half-1] + left_pad
        for i in range(inserted_num):
            # print('empty: ', half, caption_id, start+i*self.caption_window_size_frame)
            self.data.append(((game_id, half-1, start+i*self.caption_window_size_frame), (caption_id, ""), 0))
            caption_id += 1
        last_frame[half-1] = frame
        return caption_id
    
    def _cal_frame_from_time(self, time, framerate=1):
        minutes, seconds = time.split(' ')[-1].split(':')
        minutes, seconds = int(minutes), int(seconds)
        return framerate * ( seconds + 60 * minutes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            vfeats (np.array): clip of features.
            caption_tokens (np.array): tokens of captions.
            clip_id (np.array): clip id.
            caption_id (np.array): caption id.
            caption (List[strings]): list of original captions.
        """
        clip_id, (caption_id, caption), class_label = self.data[idx]
        vfeats = self.video_processor(clip_id, self.game_feats)
        caption_tokens = self.text_processor(caption)
        # load video feature
        video_id, half, frame = clip_id
        video_path = self.video_game_feats_path[video_id][half]
        position = self.position_processor(clip_id, self.game_positions)
        return video_path, vfeats, caption_tokens, clip_id[0], caption_id, caption, class_label, position
    
    def getCorpus(self, split=["train"]):
        """
        Args:
            split (string): split of dataset
        Returns:
            corpus (List[string]): vocabulary build from split.
        """
        corpus = [annotation['anonymized'] for game in getListGames(split, task="caption") for annotation in json.load(open(os.path.join(self.path, game, self.labels)))["annotations"]]
        return corpus
    
    def detokenize(self, tokens, remove_EOS=False):
        """
        Args:
            tokens (List[int]): tokens of caption
        Returns:
            caption (string): string obtained after replacing each token by its corresponding word
        """
        string = self.text_processor.detokenize(tokens)
        return string
        # return string.rstrip(f" {self.text_processor.vocab.lookup_token(EOS_TOKEN)}") if remove_EOS else string

def feats2clip(feats, stride, clip_length, padding="replicate_last", off=0):

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]

def positions2clip(positions, stride, clip_length, padding="replicate_last", off=0):
    idx = torch.arange(start=0, end=len(positions)-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, len(positions)-1).tolist()

    clip_positions = []
    for index in idx:
        clip_positions.append([positions[i] for i in index])
        
    return clip_positions


class SoccerNetClipsTesting(Dataset):
    """
    This class is used to download and pre-compute clips from the SoccerNet dataset for spotting inference phase.
    """
    def __init__(self, path, pos_path, features="baidu_soccer_embeddings.npy", split=["test"], version=2, framerate=1, window_size=15):
        self.path = path
        self.pos_path = pos_path
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        self.version = version
        self.split=split
        labels, num_classes, dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.labels = labels
        self.num_classes = num_classes
        self.dict_event = dict_event

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
            video_features1 (np.array): list of video features.
            video_features2 (np.array): list of video features.
        """

         # Load features
        if self.features == "224p_5fps.npy":
            feat_half1 = np.squeeze(np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features), allow_pickle=True).item()['features'])
            feat_half2 = np.squeeze(np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features), allow_pickle=True).item()['features'])
        else:
            feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))
        
        video_features1 = feat_half1.copy()
        video_features2 = feat_half2.copy()

        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

        # Load 2d positions
        pos_half1_path = os.path.join(self.pos_path, self.listGames[index], "1_sn_position.json")
        pos_half2_path = os.path.join(self.pos_path, self.listGames[index], "2_sn_position.json")
        pos_half1 = list(json.load(open(pos_half1_path)).values())
        pos_half2 = list(json.load(open(pos_half2_path)).values())

        # check if annoation exists
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
            labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = self.framerate * ( seconds + 60 * minutes ) 

                
                if event not in self.dict_event or half > 2:
                    continue
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = value

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)
        
        pos_half1 = positions2clip(pos_half1, stride=1, off=int(self.window_size_frame/2), clip_length=self.window_size_frame)
        pos_half2 = positions2clip(pos_half2, stride=1, off=int(self.window_size_frame/2), clip_length=self.window_size_frame)

        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2, video_features1, video_features2, pos_half1, pos_half2

    def __len__(self):
        return len(self.listGames)

class PredictionCaptions(Dataset):
    def __init__(self, SoccerNetPath, pos_path, PredictionPath, features="baidu_soccer_embeddings.npy", split=["train"], version=2, framerate=2, window_size=15):
        self.path = SoccerNetPath
        self.pos_path = pos_path
        self.PredictionPath = PredictionPath
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        self.labels, _, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.split = split

        self.data = list()
        self.game_feats = list()
        self.game_positions = list()
        self.video_game_feats_path = list()

        l_pad = self.window_size_frame//2 + self.window_size_frame%2
        r_pad = self.window_size_frame//2 

        for game_id, game in enumerate(tqdm(self.listGames)):
            
            # Load features
            feat_half1_path = os.path.join(self.path, game, "1_" + self.features)
            feat_half2_path = os.path.join(self.path, game, "2_" + self.features)
            self.video_game_feats_path.append((feat_half1_path, feat_half2_path))

            if self.features == "224p_5fps.npy":
                feat_half1 = np.squeeze(np.load(feat_half1_path, allow_pickle=True).item()['features'])
                feat_half2 = np.squeeze(np.load(feat_half2_path, allow_pickle=True).item()['features'])
            else:
                feat_half1 = np.load(feat_half1_path)
                feat_half2 = np.load(feat_half2_path)

            feat_half1 = np.pad(feat_half1.reshape(-1, feat_half1.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            feat_half2 = np.pad(feat_half2.reshape(-1, feat_half2.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")

            self.game_feats.append((feat_half1, feat_half2)) 

            # Load 2d positions
            pos_half1_path = os.path.join(self.pos_path, game, "1_sn_position.json")
            pos_half2_path = os.path.join(self.pos_path, game, "2_sn_position.json")
            pos_half1 = json.load(open(pos_half1_path))
            pos_half2 = json.load(open(pos_half2_path))
            self.game_positions.append((pos_half1, pos_half2))

            # Load labels
            preds = json.load(open(os.path.join(self.PredictionPath, game, "results_spotting.json")))
            
            for caption_id, annotation in enumerate(preds["predictions"]):

                if annotation["label"] not in self.dict_event:
                    continue

                time = annotation["gameTime"]
                half = int(time[0])
                if half > 2:
                    continue

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * ( int(seconds) + 60 * int(minutes)) 
                
                self.data.append(((game_id, half-1, frame), caption_id))
        
        # launch a VideoProcessor that will create a clip around a caption
        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)
        # launch a PositionProcessor that will create a clip
        self.position_processor = SoccerNetPositionProcessor(self.window_size_frame)
        # launch a TextProcessor that will tokenize a caption
        self.text_processor = TikTokenTextProcessor(self.getCorpus(split=["train"]))
        self.vocab_size = len(self.text_processor.vocab)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            vfeats (np.array): clip of features.
            clip_id (np.array): clip id.
            caption_id (np.array): caption id.
        """
        clip_id, caption_id = self.data[idx]
        vfeats = self.video_processor(clip_id, self.game_feats) 
        # load video features
        video_id, half, frame = clip_id
        video_path = self.video_game_feats_path[video_id][half]
        video_feature = torch.from_numpy(np.squeeze(np.load(video_path, allow_pickle=True).item()['features']))
        # video_feature = torch.from_numpy(np.load(video_path))
        position = self.position_processor(clip_id, self.game_positions)
        return video_feature, vfeats, clip_id[0], caption_id, position


    def detokenize(self, tokens, remove_EOS=True):
        """
        Args:
            tokens (List[int]): tokens of caption
        Returns:
            caption (string): string obtained after replacing each token by its corresponding word
        """
        string = self.text_processor.detokenize(tokens)
        return string
        #return string.rstrip(f" {self.text_processor.vocab.lookup_token(EOS_TOKEN)}") if remove_EOS else string
    
    def getCorpus(self, split=["train"]):
        """
        Args:
            split (string): split of dataset
        Returns:
            corpus (List[string]): vocabulary build from split.
        """
        corpus = [annotation['anonymized'] for game in getListGames(split, task="caption") for annotation in json.load(open(os.path.join(self.path, game, self.labels)))["annotations"]]
        return corpus

if __name__ == "__main__":
    raise SystemExit("dataset.py is a library module. See README.md for dataset layout and run examples.")
