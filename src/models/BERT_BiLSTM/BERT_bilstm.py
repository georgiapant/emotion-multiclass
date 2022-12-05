import torch.nn as nn
from transformers import BertModel
import torch


class BertClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, num_labels, BERT_MODEL, freeze_bert=False):

        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, num_labels

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True)

        self.LSTM = nn.LSTM(4*D_in, 512, num_layers=1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * 512, D_out)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(2*512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        hidden_states = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)["hidden_states"]

        # embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]
        pooled_output = torch.cat(tuple([attention_hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1) # concat the last four hidden layers
        lstm, _ = self.LSTM(pooled_output)
        fc_input = torch.mean(lstm, 1) # reduce the dimension of the tokens by calculating their average
        logits = self.classifier(fc_input)

        return logits