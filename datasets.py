import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class GoEmotions(Dataset):
    def __init__(self, phase='train', mode='original'):
        df = pd.read_csv(f'data/original/{phase}.tsv', sep='\t', header=None)
        print(f'{df.shape[0]} examples in the {phase} set!')
        texts = df.iloc[:, 0].tolist()
        labels = df.iloc[:, 1].tolist()
        print(labels)
        if mode == 'grouping':
            df = pd.read_csv(f'data/group/{phase}.tsv', sep='\t', header=None)
            print(f'{df.shape[0]} examples in the {phase} set!')
            texts = df.iloc[:, 0].tolist()
            labels = df.iloc[:, 1].tolist()
        elif mode == 'ekman':
            # 0 ~ 6
            df = pd.read_csv(f'data/ekman/{phase}.tsv', sep='\t', header=None)
            print(f'{df.shape[0]} examples in the {phase} set!')
            texts = df.iloc[:, 0].tolist()
            labels = df.iloc[:, 1].tolist()

        # tokenize the texts using BertTokenizer
        print('Tokenizing....')
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=128)
        input_ids = encodings['input_ids']
        attention_masks = encodings['attention_mask']
        print('Done\n\n\n')

        self.input_ids = input_ids
        self.attention_mask = attention_masks
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        # labels are like this
        # there are 28 emotions in total
        # note that this is a multi-label classification
        # each input sentence may have multiple emotions
        # 1st example --> 4,26 --> this sentence contains emotion 4 and 26
        # 2nd example --> 5 --> this sentence contains emotion 5
        # .....
        # we need to convert 4,26 to a 27-d binary vector where the 4th element and 26th element are 1
        # and elsewhere are 0

        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        label = self.labels[index]

        emotion_labels = [int(each) for each in label.split(',')]  # (4,26) --> [4, 26]
        num_labels = {'original': 28, 'grouping': 4, 'ekman': 7}[self.mode]
        label_encoding = np.zeros(num_labels)
        for each in emotion_labels:
            label_encoding[each] += 1
        label_encoding = torch.tensor(label_encoding, dtype=torch.float)

        item = {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label_encoding': label_encoding}

        return item


def loader(phase, mode, batch_size, n_workers=4):
    dataset = GoEmotions(phase, mode)
    shuffle = True if phase == 'train' else False
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, pin_memory=True)
    return data_loader
