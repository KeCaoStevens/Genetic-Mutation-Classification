library(plyr)
library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(SnowballC)
library(tm)
library(syuzhet) 
library(stringdist)
library(Matrix)
library(xgboost)
library(caret)
library(data.table)
library(kknn)
library(e1071)
library(Rtsne)
library(randomForest)
library(h2o)
distanceInd = function(x,wordl){
  return(min(stringdist(tolower(x),wordl,method="jw")))
}

textExtract = function(t){
  txt = VCorpus(VectorSource(t))
  txt = tm_map(txt, stripWhitespace)
  txt = tm_map(txt,content_transformer(gsub),pattern = "GOF",replace="gain of function",fix=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = "LOF",replace="loss of function",fix=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = "gain-of-function",replace="gain of function",ignore.case=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = "loss-of-function",replace="loss of function",ignore.case=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = "gainof function",replace="gain of function",ignore.case=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = "lossof function",replace="loss of function",ignore.case=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = "gainof-function",replace="gain of function",ignore.case=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = "lossof-function",replace="loss of function",ignore.case=TRUE)
  txt = tm_map(txt, removeWords, stopwords("english"))
  txt = tm_map(txt,content_transformer(gsub),pattern = "/",replace=" / ",ignore.case=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = '\\(',replace=' \\( ',ignore.case=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = '\\)',replace=' \\) ',ignore.case=TRUE)
  txt = tm_map(txt,content_transformer(gsub),pattern = '-mutation',replace=' - mutation',ignore.case=TRUE)
  textNew = as.data.frame(cbind(1,content(txt[[1]])))
  t1=textNew %>% unnest_tokens(sentence,V2,token="sentences") %>%
    mutate(len = 1:length(sentence))
  return(t1)
}

findWord = function(t,wl,p,l,n){
  t2 = t %>% unnest_tokens(word,sentence)
  tl = c()
  t2_1=t2
  t2_2=t2
  t2_1$ind = lapply(t2$word,distanceInd,wordl = wl) 
  t2_1 = t2_1%>% filter(ind<p)
  tl = c(tl,t2_1$len)
  if(length(tl)<1){
    t2_1=t2
    t2_1$ind = lapply(t2$word,distanceInd,wordl = c('mutation','mutant'))  
    t2_1 = t2_1%>% filter(ind<0.05)
    tl = c(tl,t2_1$len)
  }
  tl = unique(tl)
  tl2=c()
  t2_2$ind = lapply(t2$word,distanceInd,wordl = fctWord) 
  t2_2 = t2_2%>% filter(ind<0.18)
  tl2 = c(tl2,t2_2$len)
  tl2 = unique(tl2)
  tl3 = intersect(tl,tl2)
  if(length(tl3)<1){
    tl3 = tl
    t4 = ""
  }
  if(length(tl)<1){
    t4 = ""
  }
  else{
    for (i in (1:length(tl)))
      tl=c(tl,max(1,tl[i]-n):min(tl[i]+n,l))
    tl = sort(unique(tl))
    
    t3 = t[tl,]
    t4 = paste(t3$sentence,collapse="")
  }
  return(t4)
}

reduceText = function(t){
  t1 = textExtract(t[7])
  l = max(t1$len)
  #tFct = findWord(t1,fctWord,0.18,l,0)
  #tOp = findWord(t1,opWord,0.16,l,0)
  #g = tolower(as.character(t[2]))
  #tGene = findWord(t1,g,0.1,l,0)
  v = tolower(t[3])
  vl = strsplit(v," ")[[1]]
  rat = ifelse(length(vl)>1,0.18,0.14) 
  if (length(vl)==1){
    vOne = substr(v,1,1)
    vLast = substr(v,nchar(v),nchar(v))
    vOne = aaTable[which(vOne==tolower(aaTable$single)),"three"]
    vLast = aaTable[which(vLast==tolower(aaTable$single)),"three"]
    v2 = paste0(vOne,substr(v,2,nchar(v)-1),vLast)
    v2 = tolower(v2)
    vl=c(vl,v2)
  }
  if (length(vl)>1){
    vl=na.omit(unique(cbind(vl,t[[4]],t[[5]],t[[6]])))
  }
  tVar = findWord(t1,vl,rat,l,0)
  #li = c(t[4],t[5],t[6])
  #re = which(li=="")
  #li = li[-re]
  #tWord = findWord(t1,li,0.18,l,0)
  #tr = cbind(tFct,tGene,tVar,tOp,tWord)
  return(tVar)
}

transformTxt = function(t){
  txt = VCorpus(VectorSource(t))
  txt = tm_map(txt, stripWhitespace)
  txt = tm_map(txt, content_transformer(tolower))
  txt = tm_map(txt, removePunctuation)
  txt = tm_map(txt, removeWords, stopwords("english"))
  txt = tm_map(txt, stemDocument, language="english")
  txt = tm_map(txt, removeNumbers)
  return(txt)
}

tokenToTwo = function(t){
  dtm = DocumentTermMatrix(t, control = list(weighting = weightTfIdf))
  dtm = removeSparseTerms(dtm, 0.95)
  tdm = DocumentTermMatrix(t, control = list(weighting = weightTfIdf,tokenize=BigramTokenizer))
  tdm=removeSparseTerms(tdm, 0.95)
  data2 = supply[-c(3136,351,49,697,3257,3110),"Class"]
  data2 = cbind(data2,as.data.frame(as.matrix(dtm)),as.data.frame(as.matrix(tdm)))
  names(data2) = gsub(" ", "_", names(data2))
  names(data2) = gsub("function", "fct", names(data2))
  return(data2)
}

BigramTokenizer = function(x){
  unlist(lapply(ngrams(words(x),2),paste,collapse="_"),use.names=FALSE)
}

xModel = function(d,p,c,s){
  xgb_cv = xgb.cv(data = d,
                  params = p,
                  nrounds = 1000,
                  maximize = FALSE,
                  prediction = TRUE,
                  folds = c,
                  verbose=0,
                  early_stopping_rounds = 30
  )
  r = xgb_cv$best_iteration
  set.seed(s)
  xgb_model = xgb.train(data = d,
                        params = p,
                        watchlist = list(train = d),
                        nrounds = r,
                        verbose = 0
  )
  return(xgb_model)
}

xgbFeature = function(d){
  dTrain = as.matrix(subset(d,data2>-1)[-1])
  names = dimnames(dTrain)[[2]]
  y_train = Matrix(as.matrix(subset(d,data2>-1,select="data2")-1))
  dTrain0 = xgb.DMatrix(data=dTrain, label=y_train)
  set.seed(926)
  cvFoldsList = createFolds(y_train, k=5, list=TRUE, returnTrain=FALSE)
  # Params for xgboost
  param = list(booster = "gbtree",
               objective = "multi:softprob",
               eval_metric = "mlogloss",
               num_class = 9,
               eta = .2,
               gamma = 1,
               max_depth = 6,
               min_child_weight = 1,
               subsample = .7,
               colsample_bytree = .7
  )
  xgb_model = xModel(dTrain0,param,cvFoldsList,918)
  importance_matrix = xgb.importance(names,model=xgb_model)
  imp1 = importance_matrix[1:100]$Feature
  dataOut = d[imp1]
  return(dataOut)
}

multiLogloss = function(act,pred){
  return(-mean((act*log(predsCheat)+(1-act)*log(1-predsCheat))))
}
