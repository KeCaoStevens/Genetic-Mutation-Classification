import gensim
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import logging
print('start')
glove = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/DC20693/Documents/Hantao/Word2Vec/pretrain/glove.840B.300d.w2vformat.txt")
print('glove done')
glove.save_word2vec_format("glove_null", binary=True)
#nltk.download() 
text = open('C:/Users/DC20693/Desktop/gene.txt','r',encoding="utf-8")
sentence = text.read()
print('load done')
text.close()
wordnet_lemmatizer = WordNetLemmatizer()
def review_to_wordlist(review):
    words = review.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    words = [wordnet_lemmatizer.lemmatize(w) for w in words]
    return(words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer):
    translator = str.maketrans('!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~',' '*31)
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        raw_sentence = raw_sentence.translate(translator)
        " ".join(raw_sentence.split())
        raw_sentence=raw_sentence.replace("GOF","gain_of_function")
        raw_sentence=raw_sentence.replace("LOF","loss_of_function")
        raw_sentence=raw_sentence.replace("gainof function","gain_of_function")
        raw_sentence=raw_sentence.replace("lossof function","loss_of_function")
        raw_sentence=raw_sentence.replace("gainof","gain_of_function")
        raw_sentence=raw_sentence.replace("lossof","loss_of_function")
        raw_sentence=raw_sentence.replace("gain of function","gain_of_function")
        raw_sentence=raw_sentence.replace("loss of function","loss_of_function")
        #raw_sentence=raw_sentence.replace("gain function","gain_of_function")
        #raw_sentence=raw_sentence.replace("loss function","loss_of_function")
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence))
    return sentences

sentences = review_to_sentences(sentence, tokenizer)
print('tokenize done')

glove.build_vocab(sentences,update=True)
glove.train(sentences,total_examples=len(sentences),epochs=10)

glove_name = "glove_function"
glove.save(glove_name)
