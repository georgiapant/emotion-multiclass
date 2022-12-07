import argparse

from src.models.BERT.BERT_train_eval import BertSimple
from src.models.BERT_BiLSTM.BERT_bilstm_train_eval import BertBilstm
from src.models.BERT_GRU_CAPS.BERT_GRU_Caps_train_eval import BERTGruCaps
from src.models.BERT_NRC_VAD.BERT_VAD_NRC_train_eval import BertVadNrc
from src.models.CNN.CNN_train_eval import CNN
from src.models.LSTM.LSTM_train_eval import LSTM
from src.models.Traditional.Traditional_train_eval import ML

MODELS = {"BERT": BertSimple, "BERT_bilstm": BertBilstm, "BERT_bilstm_simple": BertBilstm, "BERT_gru_caps": BERTGruCaps,
          "BERT_vad_nrc": BertVadNrc, "CNN": CNN, "LSTM": LSTM, "LR": ML, "NB": ML, "SVC": ML, "RF": ML}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='BERT')
parser.add_argument('--dataset', type=str, default='ekman') #other options
parser.add_argument('--max_len', type=int, default=126)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--es', type=str, default='val_loss')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--embed_type', type=str, default='w2v_wiki') # other options - 'glove' and 'w2v'
parser.add_argument('--analyzer', type=str, default='word') # other options 'char'
parser.add_argument('--ngram_range', type=tuple, default=(1, 2))


BERT_MODEL = 'bert-base-uncased'
project_root_path = "./"

args = parser.parse_args()
model_cls = MODELS[args.model]

model = model_cls(args.dataset, args.batch_size, args.max_len, args.epochs, args.patience, BERT_MODEL,
                  args.random_seed, project_root_path, args.es, args.embed_type, args.analyzer, args.ngram_range)

model.main(args.model)
