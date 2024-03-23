from train import S2VT_model, VideoCaptioningDataset, Dataset, DataLoader, torch, sys

if __name__ == "__main__":
    data_path = sys.argv[1]
    output_file = sys.argv[2]
    opt = dict()
    opt['label'] = data_path + '/testing_label.json'
    opt['fea_path'] = data_path + '/testing_data/feat/'
    opt['vocab_path'] = './my_vocab.json'
    opt['word2index'] = './my_word2index.json'
    video_dataset_test = VideoCaptioningDataset(opt)

    opt['video_fea_dim'] = 4096
    opt['embed_dim'] = 500
    opt['vocab_len'] = 2884
    opt['dropout'] = 0
    opt['cap_len'] = 45

    s2vt_model = S2VT_model(opt)
    s2vt_model.cuda()
    data_loader = DataLoader(video_dataset_test, batch_size=1, shuffle=False)

    s2vt_model.load_state_dict(torch.load("./s2vt_model.pth"))
    s2vt_model.eval()
    with open(output_file, 'w') as file:
        for video, captions, caption_mask, video_id in data_loader:
            video = video.cuda()
            captions = captions.cuda()
            caption_mask = caption_mask.cuda()
            s2vt_model.set_test_mode()
            cap_pred = s2vt_model(video)

            test_output = torch.argmax(cap_pred, 2)
            test_captions = []
            caption_list =test_output.tolist()
            test_captions = [[video_dataset_test.index2word[str(index)] for index in caption] for caption in caption_list]

            gt_captions = []
            caption_list =captions.tolist()
            gt_captions = [[video_dataset_test.index2word[str(index)] for index in caption] for caption in caption_list]

            print("---------gt: ", ' '.join([str(element)  for element in gt_captions[0] if str(element) != '<EOS>']))
            print("---------pred: ", ' '.join([str(element) for element in test_captions[0] if str(element) != '<EOS>']))
            file.write(video_id[0] + ',' + ' '.join([str(element) for element in test_captions[0] if str(element) != '<EOS>']) + '\n')