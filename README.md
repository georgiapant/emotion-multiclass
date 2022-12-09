# Multiclass emotion classification

### Datasets
- ISEAR dataset which is published here and used in several other publications
- Emotions dataset for NLP [3]
- Mohammad et al. (2018) [4] (kaggle)
- Matta (2020) [5] (SemEval –ec 2018)
- Demszky et al. [6] (GoEmotions)

| ID |Name | Reference | Instances | Labels |
| ----------- | ----------- | ----------- | ----------- |  ----------- |
|1| ISEAR | [Klaus et al.](https://psycnet.apa.org/doiLanding?doi=10.1037%2F0022-3514.67.1.55), [link](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)| 7.6K |7 |
|2| Sentiment analysis in text (data.world)| [link](https://data.world/crowdflower/sentiment-analysis-in-text)    | 40K | 13|
|3| Emotions dataset for NLP  (Kaggle)  | [link](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) | 20K | 6 |
|4| Sem-Eval-ec 2018| [Mohammad et al.](https://aclanthology.org/S18-1001/) | 11K | 11|
|5| GoEmotions | [Demszky et al.](https://aclanthology.org/2020.acl-main.372/) | 54K | 28|

These datasets were processed and merged to create the ones used in the experiments. To select dataset in the experiments the argument `--dataset` needs to be used

| Argument | Datasets used by ID | Changes |
| ----------- | ----------- | ----------- | 
| isear | 1 | None |
| ekman  | <ul><li>`2`</li><li>`3`</li><li>`4`</li></ul> | <ul><li>Renamed: {"love":"joy", "joy":"joy", "fear":"fear", "anger":"anger", "sadness":"sadness", <br />"surprise":"surprise"} and kept instances with only one label</li> <li> renamed: "pessimism":"fear", "anticipation":"joy", "optimism":"joy", "trust":"joy"</li> <li> removed neutral and merged {"annoyance":"anger", "disapproval":"anger", "anger":"anger",<br /> "disgust":"disgust", "joy":"joy","amusement":"joy", "approval":"joy", "excitement":"joy",<br />"gratitude":"joy",  "love":"joy", "optimism":"joy", "relief":"joy", "pride":"joy", "admiration":"joy",<br />"desire":"joy", "caring":"joy", "sadness":"sadness", "disappointment":"sadness", <br /> "embarrassment":"sadness", "grief":"sadness",  "remorse":"sadness", "surprise":"surprise", <br /> "realization":"surprise", "confusion":"surprise", "curiosity":"surprise", "fear":"fear", "nervousness":"fear"} <br /> and kept instances with only one label</li></ul> | | 
| merged   | <ul><li>`1`</li><li>`2`</li><li>`3`</li></ul> | <ul><li>Kept only the instances with these classes</li><li>Renamed fear to worry and joy to happiness</li><li>Renamed fear to worry and joy to happiness, dropped all the other classes. <br />If an instance had more than one sentiment associated with it, it was also dropped</li></ul> |

### Architectures
Several different architectures were experimented with. Use the `--model` argument with the following options to experiment with them:
- `NB`: Naive bayes algorithm - features extracted using the sklearn `TfidfVectorizer`, possible to use arguments for the analyzer and the ngram range
- `LR`: Logistic regression algorithm - features extracted using the sklearn `TfidfVectorizer`, possible to use arguments for the analyzer and the ngram range
- `RF`: Random forest algorithm - features extracted using the sklearn `TfidfVectorizer`, possible to use arguments for the analyzer and the ngram range
- `SVC`: Linear Support vector classifier algorithm - features extracted using the sklearn `TfidfVectorizer`, possible to use arguments for the analyzer and the ngram range
- `CNN`: A simple CNN - features extracted using pretrained word embeddings (options explained later) 
- `LSTM`: A simple LSTM - features extracted using pretrained word embeddings (options explained later) 
- `BERT`: The `BERT` output with a simple classifier consisting of a fully connected layer followed by a relu activation function, a dropout layer and the final fully connected layer
- `BERT_bilstm_simple`: The `BERT` output stacked with a `biLSTM` layer
- `BERT_bilstm`: The last four hidden layers of the `BERT` output are concatenated and then passed through a biLSTM layer
- `BERT_gru_caps`: The hidden states of the `BERT` output are concatenated in groups of four, then they passed through GRU layers and their outputs are concatenated together. This output is then passed through two channels in parallel, a CapsNet and a set of Fully connected layers. Both of their outputs are combined using soft voting to get the final logits. 
- `BERT_vad_nrc`: `BERT - BiLSTM` including additional features extracted from the `NRC lexicon` and the `VAD lexicons`


### Additional features
- Option for early stopping based on two metrics, _validation loss_ and _macro F1_


## To perform experiments

In the main folder (emotion-multilabel) execute:
```bash
pip install -r requirements.txt
```
Then run the following command:

```python
python -m emotion_main
```

There are several possible arguments to use when executing the script such as 

* _model_ (default: `BERT`) - options: `LR`, `NB`, `RF`, `SVC`, `CNN`, `LSTM`, `BERT`, `BERT_bilstm_simple`, `BERT_bilstm`, `BERT_vad_nrc`, `BERT_gru_caps`
* _dataset_ (default: `ekman`) - options `ekman`, `isear`, `merged`
* _max_len_ (default: `126`) 
* _batch_size_ (default: `16`)
* _epochs_ (default: `10`)
* _es_ (default: `val_loss`) - Metric to check before early stopping, options `f1` and `val_loss` which is the validation loss
* _patience_ (default: `3`) - number of epochs to be patient for early stopping
* _random_seed_ (default: `43`) - to be able to reproduce the results
* _embed_type_ (default: `fasttext`) - options `fasttext`, `glove` and `w2v`
* _analyzer_ (default: `word`) - options same as the ones for the sklearn `Tfidfvectorizer`
* _ngram_range_ (default: `(1,2)` - options same as the ones for the sklearn `Tfidfvectorizer`

For example, to execute the experiment with the BERT vad nrc model with the weighted loss and the chained scheduler 
the below command needs to be executed

```python
python -m emotion_main --model BERT_vad_nrc --epochs 100 --petience 5 --es f1
```