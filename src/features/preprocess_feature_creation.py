import nrclex
import emoji
import contractions
import re
import torch
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from src.helpers import convert_to_unicode, run_strip_accents
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
import zipfile
import os
import pickle
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def fix_encoding(text):
    text = re.sub("[\x8b]", ' ', text)
    text = re.sub("[\xc2-\xf4][\x80-\xbf]+", lambda m: m.group(0).encode('latin1').decode('utf8'), text)
    text = convert_to_unicode(text.rstrip().lower())
    text = run_strip_accents(text)
    return text


def text_preprocessing(text):
    text = re.sub("[\xc2-\xf4][\x80-\xbf]+", lambda m: m.group(0).encode('latin1').decode('utf8'), text)
    text = convert_to_unicode(text.rstrip().lower())
    text = run_strip_accents(text)
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(r" +", ' ', text)
    text = re.sub(r"_", " ", text)

    text = text.lower()
    # Demojize
    text = emoji.demojize(text)

    # Expand contraction
    text = contractions.fix(text)

    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # correct some acronyms/typos/abbreviations
    x = re.sub(r"lmao", "laughing my ass off", text)
    x = re.sub(r"amirite", "am i right", x)
    x = re.sub(r"\b(tho)\b", "though", x)
    x = re.sub(r"\b(ikr)\b", "i know right", x)
    x = re.sub(r"\b(ya|u)\b", "you", x)
    x = re.sub(r"\b(eu)\b", "europe", x)
    x = re.sub(r"\b(da)\b", "the", x)
    x = re.sub(r"\b(dat)\b", "that", x)
    x = re.sub(r"\b(dats)\b", "that is", x)
    x = re.sub(r"\b(cuz)\b", "because", x)
    x = re.sub(r"\b(fkn)\b", "fucking", x)
    x = re.sub(r"\b(tbh)\b", "to be honest", x)
    x = re.sub(r"\b(tbf)\b", "to be fair", x)
    x = re.sub(r"faux pas", "mistake", x)
    x = re.sub(r"\b(btw)\b", "by the way", x)
    x = re.sub(r"\b(bs)\b", "bullshit", x)
    x = re.sub(r"\b(kinda)\b", "kind of", x)
    x = re.sub(r"\b(bruh)\b", "bro", x)
    x = re.sub(r"\b(w/e)\b", "whatever", x)
    x = re.sub(r"\b(w/)\b", "with", x)
    x = re.sub(r"\b(w/o)\b", "without", x)
    text = re.sub(r"\b(doj)\b", "department of justice", x)

    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = str(re.sub("\S*\d\S*", "", text).strip())

    return text


def preprocessing_for_bert(data, tokenizer, MAX_LEN, token_type=False):
    # Create empty lists to store outputs
    input_ids = []
    token_type_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            # pad_to_max_length=True,
            padding='max_length',  # Pad sentence to max length
            truncation='longest_first',
            return_tensors='pt',  # Return PyTorch tensor
            return_attention_mask=True,  # Return attention mask
            return_token_type_ids=token_type

        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        if token_type:
            token_type_ids.append(encoded_sent.get('token_type_ids'))

    # Convert lists to tensors
    # input_ids = torch.tensor(input_ids)
    # attention_masks = torch.tensor(attention_masks)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if token_type:
        token_type_ids = torch.cat(token_type_ids, dim=0)

    return input_ids, token_type_ids, attention_masks


def nrc_feats(input_ids, tokenizer):
    vals_corpus = []

    for sentence in input_ids:
        vals_sentence = []
        sentence = tokenizer.convert_ids_to_tokens(sentence)
        for word in sentence:
            emos = nrclex.NRCLex(word)
            frqs = emos.affect_frequencies.values()
            vals_sentence.append(list(frqs))
        vals_corpus.append(vals_sentence)
    feat = torch.tensor(vals_corpus)
    return feat


def vad_feats(input_ids, tokenizer, MAX_LEN, project_root_path):
    arousal_dict = json.load(open(project_root_path + "/data/resources/vad/arousal_dict.json"))
    valence_dict = json.load(open(project_root_path + "/data/resources/vad/valence_dict.json"))
    dom_dict = json.load(open(project_root_path + "/data/resources/vad/dom_dict.json"))

    vad_corpus = []

    for sentence in input_ids:
        vad_sentence = []
        sentence = tokenizer.batch_decode(sentence, clean_up_tokenization_spaces=False)
        arousal_vec = [arousal_dict.get(i) for i in sentence]
        arousal_vec = [float(0.5) if v is None else float(v) for v in arousal_vec]
        vad_sentence.extend(arousal_vec)

        valence_vec = [valence_dict.get(i) for i in sentence]
        valence_vec = [float(0.5) if v is None else float(v) for v in valence_vec]
        vad_sentence.extend(valence_vec)

        dom_vec = [dom_dict.get(i) for i in sentence]
        dom_vec = [float(0.5) if v is None else float(v) for v in dom_vec]
        vad_sentence.extend(dom_vec)

        vad_sentence = np.reshape(vad_sentence, (3, MAX_LEN)).T

        vad_corpus.append(vad_sentence)

    vad = torch.tensor(vad_corpus).float()
    return vad


def create_dataloaders_BERT(X, y, tokenizer, MAX_LEN, BATCH_SIZE, sampler='sequential', token_type=False):
    print('Tokenizing data...', flush=True)

    if token_type:
        inputs, token_type_ids, masks = preprocessing_for_bert(X, tokenizer, MAX_LEN, token_type=True)

        tensor_data = inputs, token_type_ids, masks
    else:
        inputs, _, masks = preprocessing_for_bert(X, tokenizer, MAX_LEN, token_type=False)
        tensor_data = inputs, masks

    if y is None:
        data = TensorDataset(*tensor_data)
    else:
        labels = torch.tensor(y)  # .values)  # .values)
        data = TensorDataset(*tensor_data, labels)

    if sampler == 'sequential':
        sampler = SequentialSampler(data)
    elif sampler == 'random':
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)
    print('Done', flush=True)
    return dataloader


def create_embedding_matrix(X_train, X_dev, model, embed_type='w2v_wiki', MAX_SEQ_LEN=100,
                            EMBED_NUM_DIMS=300, project_root_path=None):

    texts_train = [''.join(text_preprocessing(text)) for text in X_train]
    texts_dev = [''.join(text_preprocessing(text)) for text in X_dev]

    text = np.concatenate((texts_train, texts_dev), axis=0)
    text = np.array(text)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    sequence_train = tokenizer.texts_to_sequences(texts_train)
    X_train_pad = pad_sequences(sequence_train, maxlen=MAX_SEQ_LEN)

    sequence_dev = tokenizer.texts_to_sequences(texts_dev)
    X_dev_pad = pad_sequences(sequence_dev, maxlen=MAX_SEQ_LEN)

    index_of_words = tokenizer.word_index
    print('Number of unique words: {}'.format(len(index_of_words)))

    # get embeddings from file
    try:
        if embed_type == 'w2v_wiki':
            fpath = 'embeddings/wiki-news-300d-1M.vec'

            if not os.path.isfile(fpath):
                print('Downloading word vectors...')
                urllib.request.urlretrieve(
                    'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                    project_root_path + '/data/resources/wiki-news-300d-1M.vec.zip')
                print('Unzipping...')
                with zipfile.ZipFile('wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
                    zip_ref.extractall('embeddings')
                print('done.')

                # os.remove('wiki-news-300d-1M.vec.zip')

        elif embed_type == 'w2v':
            fpath = project_root_path + "/data/resources/embedding_word2vec.txt"

        elif embed_type == 'glove':
            fpath = project_root_path + "/data/resources/glove.6B." + str(EMBED_NUM_DIMS) + "d.txt"
            # pd.read_csv(r"C:\Users\georgiapant\PycharmProjects\REBECCA\Resources\glove.6B\glove.6B.100d.txt")

    except Exception as e:
        print(e)

    # create embedding matrix

    # vacab size is number of unique words + reserved 0 index for padding
    vocab_size = len(index_of_words) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, EMBED_NUM_DIMS))

    with open(fpath, encoding="utf-8") as f:

        for line in f:
            word, *vector = line.split()
            if word in index_of_words:
                idx = index_of_words[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:EMBED_NUM_DIMS]

    # # Inspect unseen words
    new_words = 0

    for word in index_of_words:
        entry = embedding_matrix[index_of_words[word]]
        if all(v == 0 for v in entry):
            new_words = new_words + 1

    print('Words found in wiki vocab: ' + str(len(index_of_words) - new_words))
    print('New words found: ' + str(new_words))

    # save tokenizer (to be used for test set)
    with open(project_root_path + '/models/' + model + '_tokenizer', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X_train_pad, X_dev_pad, vocab_size, embedding_matrix


def tokenize(text):
    data = word_tokenize(text)

    # stemming with nltk
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]

    return stem_data


def tf_idf(X_train, X_test, analyzer='word', ngrma_range=(1,2)):
    # TFIDF, unigrams and bigrams
    vect = TfidfVectorizer(tokenizer=tokenize, sublinear_tf=True, norm='l2', analyzer=analyzer, ngram_range=ngrma_range)

    # fit on our complete corpus
    vect.fit_transform(X_train)

    # transform testing and training datasets to vectors
    X_train_vect = vect.fit_transform(X_train)
    X_test_vect = vect.transform(X_test)

    return X_train_vect, X_test_vect, vect