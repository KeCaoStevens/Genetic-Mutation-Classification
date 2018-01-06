import gensim
print('start')
glove = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format("C:/Users/DC20693/Documents/Hantao/Word2Vec/pretrain/glove.840B.300d.w2vformat.txt")
print('glove done')

#glove_name = "glove_null"
#glove.save(model_name)
