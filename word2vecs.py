from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
print('start')
gg = KeyedVectors.load_word2vec_format('E:/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
print('google done')
fb = FastText.load_fasttext_format('E:/word2vec/wiki.en.bin')
print('fb done')
glove = KeyedVectors.load_word2vec_format('E:/word2vec/glove.840B.300d.w2vformat.txt')
print('all done')
