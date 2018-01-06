from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn import *
import pandas as pd
import numpy as np
# define model
def baseline_model():
    model = Sequential()
    model.add(Dense(256, input_dim=200, init='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(80, init='normal', activation='relu'))
    model.add(Dense(9, init='normal', activation="softmax"))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
train_variant = pd.read_csv("C:/Users/hanta/Desktop/Kaggle_PM/training_variants")
text = pd.read_csv('C:/Users/hanta/Desktop/vecOut2.csv')
sentence_vectors = text.as_matrix()
train_y = train_variant['Class'].values
label_encoder = LabelEncoder()
label_encoder.fit(train_y)
encoded_y = np_utils.to_categorical((label_encoder.transform(train_y)))

dummy_y = np_utils.to_categorical(encoded_y)
print(dummy_y.shape)
print('start modeling')
model = baseline_model()
#model.summary()
estimator = model.fit(sentence_vectors[0:3321],encoded_y,validation_split=0.2, epochs=10, batch_size=64)
print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1]))
y_pred = model.predict_proba(sentence_vectors[3321:])


