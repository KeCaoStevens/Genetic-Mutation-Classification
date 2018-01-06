#import gensim
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import string
import jellyfish as jf

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

def findWord(s,wl):
    wl =  [x for x in wl if not x in ['mutation','mutant','mutations','mutants']]
    wl = list(set(wl))
    translator = str.maketrans('!"#$%&\'()+,./:;<=>?@[\\]^`{|}~',' '*29)
    s=s.translate(translator)
    words = s.lower().split()
    words = [w.strip() for w in words]
    flag = 0
    for w in words:
        if match(w,wl)>0.81:
            flag += 1
    if flag < 1:
        s=''
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

def variation(v,aa = pd.read_csv("C:/Users/DC20693/Desktop/aatable.csv")):
    vl = [v]
    if ' ' in v:
        vl=v.split(' ')
    elif 'del' in v:
        if 'deletion' not in v:
            vt = v.replace('del','')
            t = 'del'+vt
            vl.append(t)
            try:
                index = list(aa['single']).index(v[0])
                vt = 'del'+aa.iloc[index,0]+vt[1:]
                vl.append(vt)
            except:
                vt = 'del'+vt
    elif 'wildtype' in v:
        vl=['wild','wildtype','wild-type']
    elif 'dup' in v:
        vt = v.replace('dup','dupt')
        vl = [v,vt]
    elif 'splice' in v:
        vl=['splice',v.replace('splice','')]
    else:
        vt = replaceAA(v)
        vl.append(vt)
    return (vl)

def replaceAA(v,aa = pd.read_csv("C:/Users/DC20693/Desktop/aatable.csv")):
    v2=v
    if (len(v)<6) & (any(str.isdigit(c) for c in v)) & (' ' not in v):
        if v[0] in list(aa['single']):
            index = list(aa['single']).index(v[0])
            v2=aa.iloc[index,0]+v[1:]
        if v[-1] in list(aa['single']):
            index = list(aa['single']).index(v[0])
            v2=v2[0:-1]+aa.iloc[index,0]
    return (v2)

def listToStr(li):
    temp = ''
    for line in li:
        for word in line:
            temp += word+" "
    return (temp)

def reduceText(take):
    count = take['ID']+1
    message = str(count) +  ' of 8989 data processed.'
    logging.info(message)
    s = take['Text']
    v = take['Variation'].lower()
    g = take['Gene'].lower()
    wl=[take['word1'],take['word2'],take['word3']]
    wl = [w for w in wl if type(w)== str]
    if 'amplify' in wl:
        wl.append('duplicate')
    operation = ["gain","loss","increase","decrease","inconclusive","activating","inactivating",
           "switch","function","activate","neutral","functional",
           "amorph","hypomorph","hypermorph","antimorph","neomorph","isomorph","activity"]
    vl = variation(v)
    sentence = tokenizer.tokenize(s.strip())
    sentence0 = []
    for sen in sentence:
        sen=replaceWords(sen)
        #round 1: normal search
        temp = findWord(sen,vl)
        temp2 = findWord(temp,operation)
        if len(temp2)>0:
            sentence0.append(toWordlist(temp2))
        else:
            sentence0.append(toWordlist(temp))
    cleaned0 = [x for x in sentence0 if len(x) > 0]
    take['Round']+=1
    if len(cleaned0)>0:
        #take['Reduced']=listToStr(cleaned0)
        take['Reduced']=cleaned0
        return (take)
    else:
        #round 2 keywords search
        for sen in sentence:
            sen=replaceWords(sen)
            digit = ''
            if (any(str.isdigit(c) for c in v)) & (v[0].isdigit()==False):
                last = 0
                for c in v[1:]:
                    if last <1:
                        if c.isdigit():
                            digit += c
                        else:
                            last += 1
            wl.append(digit)
            temp = findWord(sen,wl)
            temp2 = findWord(temp,operation)
            if len(temp2)>0:
                sentence0.append(toWordlist(temp2))
            else:
                sentence0.append(toWordlist(temp))
        cleaned0 = [x for x in sentence0 if len(x) > 0]
        take['Round']+=1
        if len(cleaned0)>0:
            #take['Reduced']=listToStr(cleaned0)
            take['Reduced']=cleaned0
            return (take)
        else:
            #round 3 function search
            for sen in sentence:
                sen=replaceWords(sen)
                temp = findWord(sen,operation)
                sentence0.append(toWordlist(temp2))
            cleaned0 = [x for x in sentence0 if len(x) > 0]
            take['Round']+=1
            #take['Reduced']=listToStr(cleaned0)
            take['Reduced']=cleaned0
            return (take)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    logging.info('start')
    table = pd.read_csv('C:/Users/dongg/Desktop/kaggle/kaggle.csv')
    table=table.drop(['Unnamed: 0'],axis=1)
    table['Round']=0
    table['Reduced']=''
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    translator = str.maketrans('!"#$%&\'()*+,-./:;<=>?@[\\]_^`{|}~',' '*32)
    stops = [s for s in stopwords.words("english") if s not in ['not','no','out','against','off','nor','down','up']]
    stops.extend(['fig','figure','author','patient','doctor','introduction','method','conclusion','discussion'])
    wordnet_lemmatizer = WordNetLemmatizer()
    table = table.apply(reduceText,axis=1)
    table.to_pickle('C:/Users/dongg/Desktop/kaggle/kaggletext.pkl')
