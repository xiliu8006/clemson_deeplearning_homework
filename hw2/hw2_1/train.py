import os
import torch
import tqdm
import re
import sys
import json
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class VideoCaptioningDataset(Dataset):
    def __init__(self, opt):
        super(VideoCaptioningDataset, self).__init__()
        self.data = pd.read_json(opt['label'])
        self.feature_path = opt['fea_path']
        self.captioning = self.data['caption']
        self.video_id = self.data['id']
        self.vocab_path = opt['vocab_path']
        self.word2index = opt['word2index']
        self.counting_caption()
        self.refining_captioning()
        self.word2index_map()
        self.load_video_feature()

    def counting_caption(self):
        if self.vocab_path is None:
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
            with open('./my_vocab.json', 'w') as f:
                json.dump(self.word_counter, f)
        else:
            with open('./my_vocab.json', 'r') as json_file:
                self.word_counter = json.load(json_file)

                
    def word2index_map(self):
        if self.word2index is None:
            self.index2word = {0: "<PAD>", 1:"<BOS>", 2:"<EOS>", 3: "<UNK>"}
            self.word2index = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
            for idx, word in enumerate(self.word_counter):
                self.index2word[idx + 4] = word
                self.word2index[word] = idx + 4
            self.vocal_len = len(self.word2index)
            with open('./my_index2word.json', 'w') as f:
                json.dump(self.index2word, f)
            with open('./my_word2index.json', 'w') as f:
                json.dump(self.word2index, f)
        else:
            with open('./my_index2word.json', 'r') as f:
                self.index2word = json.load(f)
            with open('./my_word2index.json', 'r') as f:
                self.word2index = json.load(f)

        
    
    def find_captioning_indices(self, captioning):
        indices = list()
        for word in captioning:
            indices.append(self.word2index[word])
        return indices
    
    def refining_captioning(self):
        whole_refine_captioning = list()
        max_len = 45
        for captioning_list in self.captioning:
            refined_captioning_list = list()
            for captioning in captioning_list:
                refined_captioning = ['<BOS>']
                words = re.findall(r'\w+', captioning.lower())
                # if max_len < len(words):
                #     max_len = len(words)
                for word in words:
                    if word in self.word_counter:
                        refined_captioning.append(word)
                    else:
                        refined_captioning.append('<UNK>')
                refined_captioning_list.append(refined_captioning)
            whole_refine_captioning.append(refined_captioning_list)
        self.refine_captioning = whole_refine_captioning

        padding_captioning = list()
        padding_captioning_mask = list()
        for captioning_list in self.refine_captioning:
            padding_captioning_list = list()
            captioning_list_mask = list()
            for captioning in captioning_list:
                padding_caption = captioning
                len_cap = len(captioning)
                padding_mask = np.ones(max_len)
                padding_mask[len_cap+1:] = 0
                for _ in range(max_len - len_cap):
                    padding_caption.append('<EOS>')
                captioning_list_mask.append(padding_mask)
                padding_captioning_list.append(padding_caption)
            padding_captioning_mask.append(captioning_list_mask)
            padding_captioning.append(padding_captioning_list)
        self.padding_captioning = padding_captioning
        self.padding_mask = padding_captioning_mask
    
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
        captioning_mask = self.padding_mask[index]
        captioning_indices_list = list()
        for captioning in captioning_list:
            captioning_indices = self.find_captioning_indices(captioning)
            captioning_indices_list.append(torch.tensor(captioning_indices, dtype=torch.long))
        caption_index = random.randint(0, len(captioning_indices_list)-1)   
        return torch.from_numpy(video_fea).contiguous().float(), captioning_indices_list[caption_index], torch.from_numpy(captioning_mask[0]), self.video_id[index]

class S2VT_model(torch.nn.Module):
    def __init__(self, opt):
        super(S2VT_model, self).__init__()
        self.video_fea_dim = opt['video_fea_dim']
        self.embed_dim = opt['embed_dim']
        self.word_number = opt['vocab_len']
        self.dropout_rate = opt['dropout']
        self.cap_len = opt['cap_len']
        self.linear1 = nn.Linear(self.video_fea_dim, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, self.word_number)

        self.lstm1 = nn.LSTM(self.embed_dim,  self.embed_dim, batch_first=True)
        self.lstm2 = nn.LSTM(2* self.embed_dim,  self.embed_dim, batch_first=True)

        self.embedding = nn.Embedding(self.word_number, self.embed_dim)
        self.drop = nn.Dropout(p=self.dropout_rate)
        self.training = True
        self.teaching_rate = 1
    
    def set_test_mode(self):
        self.training = False

    
    def forward(self, x, x_cap=None):
        self.teaching_rate = max(0.1, self.teaching_rate - 1/10000)
        use_teacher_forcing = True if random.random() < self.teaching_rate else False
        video_len = x.shape[1]
        x = self.linear1(self.drop(x))
        x = torch.cat((x , torch.zeros([x.shape[0], self.cap_len - 1, self.embed_dim]).cuda()), dim=1)
        output, hidden_state = self.lstm1(x)
        if use_teacher_forcing and x_cap is not None:
            x_cap = self.embedding(x_cap[:, 0:self.cap_len - 1])
            padding = torch.zeros([x.shape[0], video_len, self.embed_dim])
            padding = padding.cuda()
            x_cap = torch.cat((padding, x_cap), dim=1)
            x_fused = torch.cat((output, x_cap), dim=2)
            output, hidden_state = self.lstm2(x_fused)
            pred_cap = output[:, video_len:, :]
            pred_cap = self.drop(pred_cap)
            pred_cap = self.linear2(pred_cap)
            return pred_cap
        else:
            pred_cap = []

            padding = torch.zeros([x.shape[0], video_len, self.embed_dim]).cuda()
            x_fused = torch.cat((output[:, :video_len, :], padding), dim=2)
            encoder_output, hidden_state = self.lstm2(x_fused)

            bos_id = torch.ones(x.shape[0], dtype=torch.long).cuda()
            bos_embed = self.embedding(bos_id)
            lstm_start_input = torch.cat((output[:, video_len, :], bos_embed), 1).view(-1, 1, 2 * self.embed_dim)


            decoder_input, h_state = self.lstm2(lstm_start_input, hidden_state)
            decoder_output = self.linear2(decoder_input)
            pred_cap.append(decoder_output)


            for i in range(self.cap_len - 2):
                cur_lstm_input = torch.cat((output[:, video_len + i + 1, :], decoder_input.view(-1, self.embed_dim)), 1).view(-1, 1, 2 * self.embed_dim)
                decoder_input, h_state = self.lstm2(cur_lstm_input, h_state)
                decoder_input = self.drop(decoder_input)
                decoder_output = self.linear2(decoder_input)
                pred_cap.append(decoder_output)
                
        return torch.cat(pred_cap, dim=1)




if __name__ == "__main__":
    data_path = sys.argv[1]
    opt = dict()
    opt['label'] = data_path + '/training_label.json'
    opt['fea_path'] = data_path + '/training_data/feat/'
    opt['vocab_path'] = None
    opt['word2index'] = None
    video_dataset_train = VideoCaptioningDataset(opt)

    opt['video_fea_dim'] = 4096
    opt['embed_dim'] = 500
    opt['vocab_len'] = 2884
    opt['dropout'] = 0.2
    opt['cap_len'] = 45

    s2vt_model = S2VT_model(opt)
    s2vt_model.cuda()
    data_loader = DataLoader(video_dataset_train, batch_size=16, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(s2vt_model.parameters(), lr=0.0001)
    writer = SummaryWriter('runs/my_experiment')

    i = 0
    for epoch in range(400):
        for video, captions, caption_mask, video_id in data_loader:
            video = video.cuda()
            captions = captions.cuda()
            caption_mask = caption_mask.cuda()
            cap_pred = s2vt_model(video, captions)
            test_output = torch.argmax(cap_pred, 2)
            cap_pred = cap_pred.contiguous().view(-1, opt['vocab_len'])


            cap_gt = captions[:, 1:].contiguous().view(-1)
            cap_mask = caption_mask[:, 1:].contiguous().view(-1)

            loss = loss_func(cap_pred, cap_gt)
            loss = torch.sum(loss * cap_mask) / cap_mask.sum()



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%20 == 0:
                writer.add_scalar('Loss/train', loss.item(), i)
                print("Epoch: %d  iteration: %d , loss: %f" % (epoch, i, loss))
                
            if i%500 == 0 and i != 0:
                gt_captions = []
                caption_list =captions.tolist()
                gt_captions = [[video_dataset_train.index2word[index] for index in caption] for caption in caption_list]

                test_captions = []
                caption_list =test_output.tolist()
                test_captions = [[video_dataset_train.index2word[index] for index in caption] for caption in caption_list]

                print("---------gt: ", ' '.join(str(element) for element in gt_captions[0]))
                print("---------pred: ", ' '.join(str(element) for element in test_captions[0]))
            
            if i%2000 == 0 and i != 0:
                torch.save(s2vt_model.state_dict(), "./s2vt_model.pth")
            i += 1











