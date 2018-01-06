import gensim
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import logging
#nltk.download() 
text = open('C:/Users/DC20693/Desktop/kaggle/abstractOut.txt','r',encoding="utf-8")
sentence = text.read()
print('load done')
text.close()
wordnet_lemmatizer = WordNetLemmatizer()
stops = [s for s in stopwords.words("english") if s not in ['not','no','out','against','off','nor','down','up']]

def review_to_wordlist(review):
    words = review.lower().split()
    stops = set(stopwords.words("english"))
    words = [w.strip() for w in words]
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if not w.isdigit()]
    words = [w for w in words if not w in stops]
    words = [wordnet_lemmatizer.lemmatize(w) for w in words]
    words = [w for w in words if len(w)>2]
    return(words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer):
    translator = str.maketrans('!"#$%&\'()+,-./:;<=>?@[\\]^`{|}~',' '*30)
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
        raw_sentence=raw_sentence.replace("gain_of_function","gof")
        raw_sentence=raw_sentence.replace("loss_of_function","lof")
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence))
    return sentences

sentences = review_to_sentences(sentence, tokenizer)
print('tokenize done')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
#num_features = 200    # Word vector dimensionality                      
#min_word_count = 20   # Minimum word count                        
#num_workers = 4       # Number of threads to run in parallel
#context = 10          # Context window size                                                                                    
#downsampling = 1e-3

#model = gensim.models.word2vec.Word2Vec(sentences, workers=num_workers, 
           # size=num_features, min_count = min_word_count, iter=10,
            #window = context, sample = downsampling)
model1 = gensim.models.Word2Vec.load('C:/Users/DC20693/Desktop/200features_8989_obs')
model1.build_vocab(sentences,update=True)
model1.train(sentences,total_examples=len(sentences),epochs=10)
model_name = "200features_pubmed"
model1.save(model_name)
