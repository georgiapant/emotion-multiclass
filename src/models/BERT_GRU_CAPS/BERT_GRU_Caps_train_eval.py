import torch
from transformers import BertTokenizer
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import gc
from src.models.BERT_GRU_CAPS import BERTGruCaps
from src.data.create_dataset import GetData
from src.features.preprocess_feature_creation import create_dataloaders_BERT
from src.evaluate import evaluate
from src.train import train, predict
from src.helpers import set_seed

gc.collect()
np.seterr(divide='ignore', invalid='ignore')


class BertGruCaps:
    def __init__(self, dataset, BATCH_SIZE, MAX_LEN, EPOCHS, patience, BERT_MODEL, RANDOM_SEED, project_root_path,
                 es, embed_type, ngram_type, ngram_range):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using {} for inference".format(self.device))

        self.loss_fn = nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
        self.tokenizer.save_pretrained(project_root_path + "/models/tokenizer/")

        self.es = es
        self.EPOCHS = EPOCHS
        self.patience = patience
        self.MAX_LEN = MAX_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.BERT_MODEL = BERT_MODEL
        self.RANDOM_SEED = RANDOM_SEED
        self.project_root_path = project_root_path
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

    def initialize_model(self, train_dataloader, epochs, num_labels, BERT_MODEL, BATCH_SIZE):
        """
        Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        # Instantiate Bert Classifier
        classifier = BERTGruCaps.bertGruClassifier(num_labels, BERT_MODEL, BATCH_SIZE, freeze_bert=False)

        # Tell PyTorch to run the model on GPU
        classifier.to(self.device)

        # Create the optimizer
        optimizer = AdamW(classifier.parameters(),
                          lr=5e-5,  # Default learning rate
                          eps=1e-8  # Default epsilon value
                          )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=total_steps)

        return classifier, optimizer, scheduler

    def main(self, model_name):

        train_dataloader = create_dataloaders_BERT(self.X_train, self.y_train, self.tokenizer, self.MAX_LEN,
                                                   self.BATCH_SIZE, sampler='random')
        val_dataloader = create_dataloaders_BERT(self.X_val, self.y_val, self.tokenizer, self.MAX_LEN,
                                                 self.BATCH_SIZE, sampler='sequential')

        set_seed(42)  # Set seed for reproducibility

        bert_classifier, optimizer, scheduler = self.initialize_model(train_dataloader,
                                                                      epochs=self.EPOCHS,
                                                                      num_labels=self.num_labels,
                                                                      BERT_MODEL=self.BERT_MODEL,
                                                                      BATCH_SIZE=self.BATCH_SIZE)

        train(bert_classifier, train_dataloader, self.EPOCHS, self.es, self.patience, self.project_root_path,
              self.device, self.loss_fn, val_dataloader, optimizer, scheduler, evaluation=False)

        for batch in train_dataloader:
            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]
            model_scripted = torch.jit.trace(bert_classifier, (b_input_ids, b_attn_mask))
            torch.jit.save(model_scripted, self.project_root_path + '/models/' + model_name + '.pt')
            break

        print("Validation set", flush=True)
        probs_val = predict(self.X_val, self.MAX_LEN, self.BATCH_SIZE, self.device,
                            model_path=self.project_root_path + '/models/', model_name=model_name)
        evaluate(probs_val, self.y_val, labels=self.labels)

        print("Test set", flush=True)
        probs_test = predict(self.X_test, self.MAX_LEN, self.BATCH_SIZE, self.device,
                             model_path=self.project_root_path + '/models/', model_name=model_name)
        evaluate(probs_test, self.y_test, labels=self.labels)
