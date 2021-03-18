# Deep Learning PlayGround

Playing with some deep learning for various text proccesing task, mainly using keras and some tensorflow

## Architecture:
1. CNN
2. RNN (Bi-LSTM/GRU)
3. Attention Mechanism (with bi-LSTM)
4. Transformers (__In-Progress__)

## Task:
1. Classification (__In-progress__)
2. Sequence Tagging
3. Sequence-to-Sequence
4. Text Generation

## Demo:
- Data format is a two column csv with header:
`label,text`

- Run the demo
`python3 main.py -d {path_to_datafile} [cnn|rnn]`

- to see available option, Run
`python3 main.py --help` or `python3 main.py [cnn|rnn] --help`

In the future a jupyter notebook for each task will be added for a simple demo

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