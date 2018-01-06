#import gensim
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import logging
import string
import jellyfish as jf

aa = pd.read_csv("C:/Users/DC20693/Desktop/aatable.csv")
table = pd.read_csv('C:/Users/DC20693/Desktop/kaggle.csv')
table=table.drop(['Unnamed: 0','ID'],axis=1)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
translator = str.maketrans('!"#$%&\'()*+,-./:;<=>?@[\\]_^`{|}~',' '*32)
stops = [s for s in stopwords.words("english") if s not in ['not','no','out','against','off','nor','down','up']]
stops.extend(['fig','figure','author','patient','doctor','introduction','method','conclusion','discussion'])
wordnet_lemmatizer = WordNetLemmatizer()
def match(s,l):
    score = []
    for li in l:
        score.append(jf.jaro_distance(li,s))
    return (max(score))
    

def toWordlist(s):
    translator = str.maketrans('!"#$%&\'()+,-.*/:;<=>?@[\\]_^`{|}~',' '*32)
    s=s.translate(translator)
    s=s.lower()
    s=replaceWords(s)
    s.replace('gain of function','gof')
    s.replace('loss of function','lof')
    words = s.split()
    words = [w.strip() for w in words]
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if not w.isdigit()]
    words = [w for w in words if not w in stops]
    words = [wordnet_lemmatizer.lemmatize(w) for w in words]
    words = [w for w in words if len(w)>1]
    return(words)

#return  sentences when variation is found
def findVar(s,v,wl,t=aa):
    translator = str.maketrans('!"#$%&\'()+,./:;<=>?@[\\]^`{|}~',' '*29)
    s=s.translate(translator)
    if 'amplify' in wl:
        wl.append('duplicate')
    wl = [w for w in wl if type(w)=='str']
    words = s.lower().split()
    words = [w.strip() for w in words]
    flag = 0
    vlist = [v]
    for char in [' ', '_','del','dup','fs','ins']:
        if char in v:
            for vt in v.split(char):
                vlist.append(vt)
            for vt in wl:
                vlist.append(vt)
    
    v2=v
    if (len(v)<6) & (any(str.isdigit(c) for c in v)) & (' ' not in v):
        if v[0] in list(aa['single']):
            index = list(aa['single']).index(v[0])
            v2=aa.iloc[index,0]+v[1:]
        if v[-1] in list(aa['single']):
            index = list(aa['single']).index(v[0])
            v2=v2[0:-1]+aa.iloc[index,0]
        vlist.append(v2)
    vlist = [x for x in vlist if not x in ['mutation','mutant','mutations','mutants']]
    for w in words:
        if match(w,vlist)>0.81:
            flag += 1
    if flag < 1:
        vlist.append(wl)
        for w in wl:
            if match(w,vlist)>0.81:
                flag += 1
        if flag<1:
            s=''
    return (s)
        
#return sentences when mutation is found    
def findMut(s):
    l = ["gain","loss","increase","decrease","inconclusive","activating","inactivating",
           "switch","function","activate","neutral",
           "amorph","hypomorph","hypermorph","antimorph","neomorph","isomorph","activity"]
    translator = str.maketrans('!"#$%&\'()+,./:;<=>?@[\\]^`{|}~',' '*29)
    words = s.lower().split()
    words = [w.translate(translator) for w in words]
    words = [w.strip() for w in words]
    flag = 0
    for w in words:
        if match(w,l)>0.81:
            flag += 1
    if flag < 1:
        s = ''
    return (s)

def replaceWords(sen):
    sen = sen.replace("n't"," not ")
    sen = sen.replace("GOF","gain of function")
    sen = sen.replace("LOF","loss of function")
    sen = sen.replace("gainof function","gain of function")
    sen = sen.replace("lossof function","loss of function")
    sen = sen.replace("gainof","gain of function")
    sen = sen.replace("lossof","loss of function")
    sen = sen.replace("gain of function","gain of function")
    sen = sen.replace("loss of function","loss of function")
    return (sen)

def loadSentence(take):
    s = take['Text']
    v = take['Variation'].lower()
    g = take['Gene'].lower()
    wl=[take['word1'],take['word2'],take['word3']]
    sentence = tokenizer.tokenize(s.strip())
    sentences0 = []
    sentences1 = []
    for sen in sentence:
        sen=replaceWords(sen)
        temp = findVar(sen,v,wl)
        temp2 = findMut(temp)
        if len(temp)>0:
            sentences1.append(toWordlist(temp))      
        if len(temp2)>0:
            sentences0.append(toWordlist(temp2)) 
    cleaned1 = [x for x in sentences1 if len(x) > 0]
    cleaned0 = [x for x in sentences0 if len(x) > 0]
    if len(cleaned0) == 0:
        return (cleaned1)
    else:
        if len(cleaned0)!=0:  
            return (cleaned0)
        else:
            wl.append('function')
            for sen in sentence:
                temp = findVar(sen,v,wl)
                temp2 = findMut(temp)
                if len(temp)>0:
                    sentences1.append(toWordlist(temp))      
                if len(temp2)>0:
                    sentences0.append(toWordlist(temp2)) 
            cleaned1 = [x for x in sentences1 if len(x) > 0]
            cleaned0 = [x for x in sentences0 if len(x) > 0]
            if len(cleaned0) == 0:
                return (cleaned1)
            else:
                return (cleaned0)

#table['Reduced']=[loadSentence(table.iloc[i]) for i in range(0,8989)]
#table.to_pickle('E:/kaggle 2nd/text2.pkl')
#table = pd.read_pickle('E:/kaggle 2nd/text.pkl')

