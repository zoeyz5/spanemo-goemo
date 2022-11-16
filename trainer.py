import torch
import torch.nn as nn
from datasets import loader
from models.bertemo import BertEMO
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class Trainer:
    def __init__(self, args):
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device_count = torch.cuda.device_count()
        print(f"Let's use {device_count} GPUs!\n")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Loading data
        print('#' * 10 + 'Train set' + '#' * 10)
        train_laoder = loader('train', args.mode, args.batch_size, args.n_workers)
        print('#' * 10 + 'Dev set' + '#' * 10)
        dev_laoder = loader('dev', args.mode, args.batch_size_eval, args.n_workers)
        print('#' * 10 + 'Test set' + '#' * 10)
        test_laoder = loader('test', args.mode, args.batch_size_eval, args.n_workers)

        # Prepare model
        model = BertEMO(args.model)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.args = args
        self.device = device
        self.train_loader = train_laoder
        self.dev_loader = dev_laoder
        self.test_loader = test_laoder
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self):
        best_epoch = 0
        best_epoch_macro_f1 = 0
        for epoch in range(self.args.epochs):
            print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
            loss = self.train_epoch()
            f1s, precisions, recalls = self.eval()
            macro_f1 = f1s.mean()
            macro_p = precisions.mean()
            macro_r = recalls.mean()

            if macro_f1 > best_epoch_macro_f1:
                best_epoch = epoch
                best_epoch_macro_f1 = macro_f1
                torch.save(self.model.state_dict(), 'ckp/best_model.pt')

            print(f'Epoch {epoch + 1}\t'
                  f'Train Loss: {loss:.3f}\tVal Macro F1: {macro_f1:.3f}\t'
                  f'Val Macro Precision: {macro_p:.3f}\t'
                  f'Val Macro Recall: {macro_r:.3f}\n'
                  f'Best Epoch: {best_epoch + 1}\tBest Epoch Val Macro F1: {best_epoch_macro_f1:.3f}\n\n\n')
        print('Training Finished!')

        print('Loading the best ckp....')
        ckp = torch.load('ckp/best_model.pt')
        self.model.load_state_dict(ckp)
        print('Done\n')

        print('Testing the model')
        f1s, precisions, recalls = self.eval('test')
        macro_f1 = f1s.mean()
        macro_p = precisions.mean()
        macro_r = recalls.mean()
        for i in range(27):
            f1s[i] = round(f1s[i], 3)
            precisions[i] = round(precisions[i], 3)
            recalls[i] = round(recalls[i], 3)
        print(
              f'Test Macro F1: {macro_f1:.3f}\t'
              f'Test Macro Precision: {macro_p:.3f}\t'
              f'Test Macro Recall: {macro_r:.3f}\n'
              f'Test F1s: {f1s}\n'
              f'Test Precisions: {precisions}\n'
              f'Test Recalls: {recalls}\n')

    def train_epoch(self):
        self.optimizer.zero_grad()
        self.model.train()
        epoch_loss = 0
        interval = max(len(self.train_loader) // 20, 1)
        for i, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label_encoding'].to(self.device)
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if i % interval == 0 or i == len(self.train_loader) - 1:
                print(f"Batch: {i + 1}/{len(self.train_loader)}\tloss: {loss.item():.3f}")
        return epoch_loss / len(self.train_loader)

    def eval(self, phase='dev'):
        loader = self.dev_loader if phase == 'dev' else self.test_loader
        self.model.eval()
        logits_ = []
        labels_ = []
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label_encoding']
                logits = self.model(input_ids, attention_mask)
                logits = logits.gt(self.args.threshold).to(torch.long)
                logits_.append(logits.detach().to('cpu').numpy())
                labels_.append(labels.to(torch.long).numpy())

        logits_ = np.concatenate(logits_, axis=0)  # each row is a sample, each column is the pred label of the corresponding emotion
        labels_ = np.concatenate(labels_, axis=0)  # each row is a sample, each column is the ground truth label of the corresponding emotion

        f1s = []
        precisions = []
        recalls = []
        for i in range(27):
            f1 = f1_score(labels_[:, i], logits_[:, i], average='macro')
            p = f1_score(labels_[:, i], logits_[:, i], average='macro')
            r = f1_score(labels_[:, i], logits_[:, i], average='macro')

            f1s.append(f1)
            precisions.append(p)
            recalls.append(r)

        f1s = np.array(f1s)
        precisions = np.array(precisions)
        recalls = np.array(recalls)

        return f1s, precisions, recalls
