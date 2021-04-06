# Deep Learning PlayGround

Playing with some deep learning for various text proccesing task, mainly using keras and some tensorflow

## Architecture:
1. Convolutional Neural Network (CNN)
2. Recurrent Neural Network (RNN) (Bi-LSTM/bi-GRU)
3. Attention Mechanism with RNN
4. Transformers
5. Hierarchical Attention Network (HAN) (**Classifier Only**)
6. Recurrent Convolutional Neural Networks (RCNN) (**Classifier Only**)

## Task:
1. Classification
2. Sequence Labelling (**In-Progress**)
3. Sequence-to-Sequence
4. Text Generation

## Demo:
- Data format is a two column csv with header:
`label,text`

- Run the demo
`python3 main.py -d {path_to_datafile} [cnn|rnn|transformer|han|rcnn]`

- to see available option, Run
`python3 main.py --help` or `python3 main.py [cnn|rnn|transformer|han|rcnn] --help`

There is jupyter notebook available for each task (so far only classification) that both provides detailed explanation of the usage

## Data
There are parser for several dataset for each tasks. Results and configurations will be added later

### Classification

#### Emotion Dataset (id)
This dataset is consist of 4403 tweet, each labeled with one of 5 emotion (angry, joy, sadness, fear, and love)
Source : [Repo](https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset), [Paper](https://doi.org/10.1109/IALP.2018.8629262)

#### News Category (id)
This dataset is taken from IndoSum which is an Indonesian news summarization dataset. However, there is a category label which we could use as label for classification. each news is belong to one of 5 categories (tajuk utama / headline, hiburan / entertaiment, teknologi / technology, olah raga / sport, and showbiz). There are total of 18774 tokenized news dataset
Source : [Repo](https://github.com/kata-ai/indosum), [Paper](10.1109/IALP.2018.8629109)

#### IMDb (en)
This is the IMDb review dataset, commonly used for sentiment analysis (binar classification of positive and negative). There are 25k review for testing and 25k for training.

Source : [Website](https://ai.stanford.edu/~amaas/data/sentiment/) [Paper](https://www.aclweb.org/anthology/P11-1015/)

```
for this dataset only sent the directory path of the train/test to -d argument
```

#### LIAR (en)
This dataset consist of fact-check verdict of a news statement politifact. There are 12519 statement, each is given one of six verdict true, mostly-true, half-true, barely-true, false, and pants-fire.

Source : [Website](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) [Paper](https://www.aclweb.org/anthology/P17-2067.pdf)

#### AG News Dataset (en)
This dataset consists of 120000 news dataset for training and 7600 for testing. Each news is belong to one of 4 category "world", "sports", "business", and "science".

Source : [Kaggle](https://www.kaggle.com/amananandrai/ag-news-classification-dataset) [Website](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) [Paper](http://www.ra.ethz.ch/cdstore/www2005/docs/p97.pdf)

### Sequence Labelling

#### IDN Tagged Corpus (id)

#### Universal Dependencies ID (id)

#### NER (id)

#### Universal Dependencies EN (en)

#### NER (en)

## Dependencies
- tensorflow==2.4.1
- Keras==2.4.3
- numpy==1.19.5
- gensim==3.8.3
- scikit-learn==0.20.3
- sklearn-crfsuite==0.3.6
- nltk==3.4.5

## Resources:
### Paper
- CNN Word Level [Yoon K., 2014](https://www.aclweb.org/anthology/D14-1181/)
- CNN Character Level [Zhang X., 2016](https://arxiv.org/abs/1509.01626)
- RNN Additive Attention [Bahdanau D., 2015](https://arxiv.org/abs/1409.0473)
- RNN Multiplicative Attention [Luong M., 2015](https://arxiv.org/abs/1508.04025)
- RNN Self Attention [Lin Z., 2016](https://arxiv.org/abs/1703.03130)
- Transformer [Vaswani A., 2017](https://arxiv.org/abs/1706.03762)
- HAN [Yang Z., 2016](https://www.aclweb.org/anthology/N16-1174/)
- RCNN [Lai S., 2015](https://dl.acm.org/doi/10.5555/2886521.2886636)

### Web Articles
- [CNN classification 1](https://cezannec.github.io/CNN_Text_Classification/): More thorough explanation 
- [CNN classification 2](https://towardsdatascience.com/cnn-sentiment-analysis-1d16b7c5a0e7): A bit more direct hands on implementation
- [CNN character level classification](https://towardsdatascience.com/character-level-cnn-with-keras-50391c3adf33): Implementation example of Zhang X., 2016
- [RNN Attention 1](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/): Really great explanation + visualization on the matter
- [RNN Attention 2](https://blog.floydhub.com/attention-mechanism/): Explanation + comparison between Additive vs. Multiplicative. Also PyTorch implementation
- [RNN Attention 3](https://towardsdatascience.com/create-your-own-custom-attention-layer-understand-all-flavours-2201b5e8be9e): RNN Attention and classification
- [Transformer](https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51): PyTorch Implementation

### Other Great Repo
- [Text classification with Deep Learning](https://github.com/TobiasLee/Text-Classification)
- [Attention Mechanism 1](https://github.com/uzaymacar/attention-mechanisms)
- [Attention Mechanism 2](https://github.com/philipperemy/keras-attention-mechanism)

## Others:
### CRF Tokenizer
- The implementation is on `CRFTokenizer` class in `common/tokenization.py` a demo will be provided in the future
- Reference to [Barik A., 2020](https://www.aclweb.org/anthology/D19-5554/) and [the git repo](https://github.com/seelenbrecher/code-mixed-normalization)

```
Input   : Text
Label   : BIO-Tag for each character in the text
Example :

    Input = "hello world!"
    Label = "BIIIIOBIIIIB"
    Token = ["hello", "world", "!"]
``` 

- Run `train(data, label)` on CRFTokenizer
- Save model with `save_model(path_to_dir)` for future use
- Use `tokenize(text)` to tokenize text (make sure the CRF model is trained/loaded)
- Load model with `load_model(path_to_dir)`

Question or other inquiries to ajmalprayoga@gmail.com