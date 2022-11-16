import torch
import torch.nn as nn


class BertEMO(nn.Module):
    def __init__(self, mode='original'):
        super(BertEMO, self).__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # n_layers = 12
        # if nfinetune > 0:
        #     for param in self.bert.pooler.parameters():
        #         param.requires_grad = True
        #     for i in range(n_layers-1, n_layers-1-nfinetune, -1):
        #         for param in self.bert.encoder.layer[i].parameters():
        #             param.requires_grad = True
        num_labels = {'original': 28, 'grouping': 4, 'ekman': 7}[mode]
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        embs = outputs.pooler_output
        logits = self.fc(embs)
        return logits
