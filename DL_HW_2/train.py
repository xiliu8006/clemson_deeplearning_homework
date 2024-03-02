import os
import torch
import tqdm
import re
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset
import torch.nn as nn
from argparse import ArgumentParser
from collections import Counter


class VideoCaptioningDataset(Dataset):
    def __init__(self, opt):
        super(VideoCaptioningDataset, self).__init__()
        self.data = pd.read_json(opt['label'])
        self.feature_path = opt['fea_path']
        self.captioning = self.data['caption']
        self.video_id = self.data['id']
        self.counting_caption()
        self.refine_captioning()
        self.word2index_map()
        self.load_video_feature()

    def counting_caption(self):
        self.word_counter = {}
        for captioning_list in self.captioning:
            for captioning in captioning_list:
                words = re.findall(r'\w+', captioning.lower())
                for word in words:
                    if word in self.word_counter:
                        self.word_counter[word] += 1
                    else:
                        self.word_counter[word] = 1
        self.word_counter['<PAD>'] = 3
        self.word_counter['<BOS>'] = 3
        self.word_counter['<EOS>'] = 3
        self.word_counter['<UNK>'] = 3

        for key, value in list(self.word_counter.items()):
            if value < 3:
                del self.word_counter[key]
                
    def word2index_map(self):
        self.index2word = {0: "<PAD>", 1:"<BOS>", 2:"<EOS>", 3: "<UNK>"}
        self.word2index = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for idx, word in enumerate(self.word_counter):
            self.index2word[idx + 4] = word
            self.word2index[word] = idx + 4
        self.vocal_len = len(self.word2index)
    
    def find_captioning_indices(self, captioning):
        indices = list()
        for word in captioning:
            indices.append(self.word2index[word])
        return indices
    
    def refine_captioning(self):
        whole_refine_captioning = list()
        for captioning_list in self.captioning:
            refined_captioning_list = list()
            for captioning in captioning_list:
                refined_captioning = ['<BOS>']
                words = re.findall(r'\w+', captioning.lower())
                for word in words:
                    refined_captioning.append(word)
                else:
                    refined_captioning.append('<UNK>')
                refined_captioning_list.append(refined_captioning)
            whole_refine_captioning.append(refined_captioning_list)
        self.refine_captioning = whole_refine_captioning
    
    def load_video_feature(self):
        self.video_fea = list()
        for video_id in self.video_id:
            video_fea = np.load(os.path.join(self.feature_path, video_id + '.npy'))
            self.video_fea.append(video_fea)

    def __len__(self):
        return len(self.refine_captioning)

    def __getitem__(self, index):
        video_fea = self.video_fea[index]
        captioning_list = self.refine_captioning[index]
        one_hot_list = list()
        for captioning in captioning_list:
            captioning_indices = self.find_captioning_indices(captioning)
            one_hot_coding = torch.nn.functional.one_hot(captioning_indices.to(torch.int64), num_classes=self.vocal_len)
            one_hot_list.append(one_hot_coding)
        # if do not consider search beam, we will only ouput the first captioning

class S2VT_model(torch.nn.Module):
    def __init__(self, opt):
        super(S2VT_model, self).__init__()
        self.video_fea_dim = opt['video_fea_dim']
        self.embed_dim = opt['embed_dim']
        self.one_hot_dim = opt['one_hot_dim']
        self.dropout_rate = opt['dropout']
        self.cap_len = opt['cap_len']
        self.linear1 = nn.Linear(self.video_fea_dim, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.lstm1 = nn.LSTM( self.embed_dim,  self.embed_dim, batch_first=True)
        self.lstm2 = nn.LSTM(2* self.embed_dim,  self.embed_dim, batch_first=True)

        self.embedding = nn.Embedding(self.one_hot_dim, self.embed_dim)
        self.drop = nn.Dropout(p=self.dropout_rate)
        self.training = True
    
    def set_test_mode(self):
        self.training = False

    
    def forward(self, x, x_cap):

        x = self.linear1(x)
        x = self.drop(x)
        x = torch.cat((x , torch.zeros([x.shape[0], self.cap_len, self.embed_dim]).to(x.device)), dim=1)
        output, hidden_state = self.lstm1(x)

        if self.training:
            x_cap = self.embedding(x_cap)
            x_cap = torch.cat((torch.zeros([x.shape[0], x.shape[1], self.embed_dim]), x_cap), dim=1)
            x_fused = torch.cat((output, x_cap), dim=2)
            output, hidden_state = self.lstm2(x_fused)
            pred_cap = output[:, x.shape[1]:, :]
            pred_cap = self.linear2(pred_cap)
            return pred_cap
        else:
            padding = torch.zeors([x.shape[0], self.cap_len, self.embed_dim]).to(x.device)
            x_cap = torch.cat((padding, output, ))
            
            

        
    
opt = dict()
opt['label'] = '/home/xi/code/DATASET/MLDS_hw2_1_data/MLDS_hw2_1_data/training_label.json'
opt['fea_path'] = "/home/xi/code/DATASET/MLDS_hw2_1_data/MLDS_hw2_1_data/training_data/feat/"
video_dataset = VideoCaptioningDataset(opt)
print(len(video_dataset))








