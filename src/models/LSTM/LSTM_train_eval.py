import gc
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from src.data.create_dataset import GetData
from src.features.preprocess_feature_creation import create_dataloaders_BERT
from src.helpers import format_time, set_seed
# from src.models.LSTM.LSTM_model import lstm
from src.train_tf import train, predict

from src.evaluate import evaluate


gc.collect()
np.seterr(divide='ignore', invalid='ignore')


class LSTM:

    def __init__(self, dataset, BATCH_SIZE, MAX_LEN, EPOCHS, patience, BERT_MODEL, RANDOM_SEED,
                 project_root_path, es, word_embd_type, ngram_type, ngram_range):

        # self.device = torch.device('cuda')
        # print('GPU:', torch.cuda.get_device_name(0))

        # self.loss_fn = nn.CrossEntropyLoss()
        # self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
        # self.tokenizer.save_pretrained(project_root_path + "/models/tokenizer_simple/")

        self.es = es
        self.EPOCHS = EPOCHS
        self.patience = patience
        self.MAX_LEN = MAX_LEN
        self.BATCH_SIZE = BATCH_SIZE
        # self.BERT_MODEL = BERT_MODEL
        self.RANDOM_SEED = RANDOM_SEED
        self.project_root_path = project_root_path
        self.word_embd_type = word_embd_type
        create_dataset = GetData(self.project_root_path, self.RANDOM_SEED)

        if dataset == 'ekman':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.labels \
                = create_dataset.ekman()
        elif dataset == 'isear':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.labels \
                = create_dataset.isear()
        elif dataset == 'merged':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.labels \
                = create_dataset.merged()
        else:
            print("No dataset with the name {}".format(dataset))

        self.num_labels = len(self.labels)

    def main(self, model_name):

        if self.word_embd_type == 'w2v-wiki':
            EMBED_NUM_DIMS = 300
        else:
            EMBED_NUM_DIMS = 100

        t0 = time.time()

        # model = cnn()
        train(self.X_train, self.y_train, self.X_val, self.y_val, model_name, self.es, self.patience, self.EPOCHS,
              self.BATCH_SIZE, self.word_embd_type, EMBED_NUM_DIMS, self.MAX_LEN, self.num_labels, self.project_root_path)

        predict(self.X_test, self.y_test, self.labels, model_name, self.project_root_path, EMBED_NUM_DIMS)

        print(f"Total training and prediction time: {format_time(time.time() - t0)}", flush=True)
