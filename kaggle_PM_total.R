#####################################
#model diagram                      #
#####################################

          #############################    ################################     #############
######    #first-layer                #    #second-layer                  #     #third-layer#
#data#--->#train_data(70~80% training)#--->#ensemble data(30~20% training)# --->#test data  #
######    #############################    ################################     #############
          #tsne, original+feature_imp#     #prob of each class#                 #ensemble#
          ############################     ####################                 ##########
          #xgb                             #knn                                 #label of test
          #rf                              #xgb
          #svm                             #lightGBM
          #lightGBM
          #NN(*)

##################################
#data                            #
##################################
trainVariant = read.csv("C:/Users/DC20693/Desktop/kaggleMutation/training_variants")
testVariant = read.csv("C:/Users/DC20693/Desktop/kaggleMutation/test_variants")
#train
train_text = do.call(rbind,strsplit(readLines("C:/Users/DC20693/Desktop/kaggleMutation/training_text"),"||",fixed=T))
train_text = as.data.table(train_text)
train_text = train_text[-1,]
colnames(train_text) = c("ID", "Text")
train_text$ID = as.numeric(train_text$ID)
train = merge(trainVariant,train_text,by="ID")
#test
test_text = do.call(rbind,strsplit(readLines("C:/Users/DC20693/Desktop/kaggleMutation/test_text"),"||",fixed=T))
test_text = as.data.table(test_text)
test_text = test_text[-1,]
colnames(test_text) = c("ID", "Text")
test_text$ID = as.numeric(test_text$ID)
test = merge(testVariant,test_text,by="ID")


test$Class <- -1
data = rbind(train,test)
supply = read.csv("C:/Users/DC20693/Desktop/WORDLIST.csv")

#replace one to three amino acid name
#remove data point,3321-6=3315
data=data[-c(3136,351,49,697,3257,3110),]
dataText = cbind(supply[,c(1,2,3,16,17,18)],data[,"Text"])
names(dataText)=c("ID","Gene","Variation","word1","word2","word3","Text")
dataText = dataText[-c(3136,351,49,697,3257,3110),]
#ngrams
#geneList = tolower(unique(as.character(data$Gene)))
#specialWord = unique(stemDocument(
  #c("gain","loss","inactivating","activating","switch","mutant","mutation",
   # "inconclusive","variant","alteration","variation",
   # "exon","missense","silent","nonsense","delete","duplicate","amplifi",
   # "insert","substitution","truncate","fusion","frameshift",
   # "oncogenic","oncogene")))
fctWord =c("gain","loss","inconclusive","activating","inactivating","switch","function",
           "amorph","hypomorph","hypermorph","antimorph","neomorph","isomorph","activity")
opWord = c("alteration","variation","exon","missense","silent","nonsense","delete",
           "duplicate","amplifi","insert","substitution","truncate","fusion","frameshift")
dataText$word1=as.character(dataText$word1)
dataText$word2=as.character(dataText$word2)
dataText$word3=as.character(dataText$word3)
textOut = t(as.matrix(apply(dataText,1,reduceText)))
txtOut=as.data.frame(textOut)
names(txtOut)=c("TxtFct","TxtG","TxtVar","TxtOp","TxtWord")
txtOut$TxtVar=NULL
cat("TF-IDF")
#Function
txtFct = transformTxt(txtOut[,1])
dataFct = tokenToTwo(txtFct)
#Gene
txtGene = transformTxt(txtOut[,2])
dataGene = tokenToTwo(txtGene)
#Operation*
txtOP = transformTxt(txtOut[,3])
dataOP = tokenToTwo(txtOP)
#Words
txtWords = transformTxt(txtOut[,4])
dataWords = tokenToTwo(txtWords)

#Sentiment analysis
sentiment = get_nrc_sentiment(data$Text) 

#feature importance w/ xgb
dataFct_f = xgbFeature(dataFct)
dataGene_f = xgbFeature(dataGene)
dataOP_f = xgbFeature(dataOP)
dataWords_f = xgbFeature(dataWords)

#test
dataFct_f=dataFct_f[1:50]
names(dataFct_f)=paste(names(dataFct_f),"_f")
dataGene_f=dataGene_f[1:50]
names(dataGene_f)=paste(names(dataGene_f),"_g")
dataOP_f=dataOP_f[1:50]
names(dataOP_f)=paste(names(dataOP_f),"_o")
dataWords_f=dataWords_f[1:50]
names(dataWords_f)=paste(names(dataWords_f),"_w")
##################################################################

#important features
names = dimnames(train_sparse4)[[2]]
importance_matrix = xgb.importance(names,model=xgb_model)
imp1 = importance_matrix[1:50]$Feature


#t-snes
tsne_f = Rtsne(dataFct_f,dims=2,verbose = TRUE,check_duplicates = FALSE,theta=0.0,max_iter=1200)
tsne_g = Rtsne(dataGene_f,dims=2,verbose = TRUE,check_duplicates = FALSE,theta=0.0,max_iter=1200)
tsne_o = Rtsne(dataOP_f,dims=2,verbose = TRUE,check_duplicates = FALSE,theta=0.0,max_iter=1200)
tsne_w = Rtsne(dataWords_f,dims=2,verbose = TRUE,check_duplicates = FALSE,theta=0.0,max_iter=1200)

data_tsne = cbind(data[-c(3136,351,49,697,3257,3110),"Class"],tsne_f$Y,tsne_g$Y,tsne_o$Y,tsne_w$Y)
data_tsne = as.data.frame(data_tsne)
names(data_tsne)=c("Class","F1","F2","G1","G2","O1","O2","W1","W2")

#split
set.seed(7282017)
layer2index=sample(1:3321,300)
layer1train = data[-layer2index,]
layer2train = predict(model,newdata=data[layer2index,])
layer2label = data[layer2index,4]

###################################
#model                            #
###################################
#xgb
#full features
dataFinal = cbind(dataFct_f,dataGene_f,dataOP_f,dataWords_f)
dataFinal$Class=data[-c(3136,351,49,697,3257,3110),"Class"]
names(dataFinal)=gsub(" ", "", names(dataFinal))
#train
dTrain = as.matrix(subset(dataFinal,Class>-1)[-201])
y_train = Matrix(as.matrix(subset(dataFinal,Class>-1,select="Class")-1))
dTrain0 = xgb.DMatrix(data=dTrain, label=y_train)
#test
dTest =  as.matrix(subset(dataFinal,Class==-1)[-201])
dtest0 = xgb.DMatrix(data=dTest)
#importance (xgb)
set.seed(957)
cvFoldsList = createFolds(y_train, k=5, list=TRUE, returnTrain=FALSE)
# Params for xgboost
param = list(booster = "gbtree",
             objective = "multi:softprob",
             eval_metric = "mlogloss",
             num_class = 9,
             eta = .1,
             gamma = 1,
             max_depth = 6,
             min_child_weight = 1,
             subsample = .7,
             colsample_bytree = .7
)

xgb_model=xModel(dTrain0,param,cvFoldsList,8117)
test_ids = subset(data,Class==-1,ID)
cat("Predictions")
preds = as.data.table(t(matrix(predict(xgb_model, dtest0), nrow=9, ncol=nrow(dtest0))))
colnames(preds) = c("class1","class2","class3","class4","class5","class6","class7","class8","class9")
write.table(data.table(ID=test_ids, preds), "C:/Users/DC20693/Desktop/submission81_1.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)

#xgb tsne
#train
dTrain_T = as.matrix(subset(data_tsne,Class>-1)[-1])
dTrain_T_0 = xgb.DMatrix(data=dTrain_T, label=y_train)
#test
dTest_T =  as.matrix(subset(data_tsne,Class==-1)[-1])
dtest_T_0 = xgb.DMatrix(data=dTest_T)
xgb_model_tsne = xModel(dTrain_T_0,param,cvFoldsList,8117)
preds = as.data.table(t(matrix(predict(xgb_model_tsne, dtest0), nrow=9, ncol=nrow(dtest_T_0))))
colnames(preds) = c("class1","class2","class3","class4","class5","class6","class7","class8","class9")
write.table(data.table(ID=test_ids, preds), "C:/Users/DC20693/Desktop/submission81_2.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)

#svm,radical:tsne
c = seq(0.0001,1,length.out = 40)
g = seq(0.0001,3,length.out = 40)
svmCtr = tune.control(nrepeat = 3, repeat.aggregate = mean, 
                      sampling= "fix",sampling.aggregate = mean,cross = 5,
                      best.model = TRUE, performances = TRUE, error.fun = NULL)
set.seed(81)    
dTrainSVM = as.data.frame(dTrain)
dSVM_T = subset(data_tsne,Class>-1)
dSVM_T$Class=factor(dSVM_T$Class)
dTrainSVM$Class = factor(data[-c(3136,351,49,697,3257,3110),4][1:3315])
svmObj1 = tune.svm(Class~., data = dSVM_T[sample(1:3315,500),], scale=TRUE,
                   kernel="radial",
                   cost = c,
                   gamma = g,
                   tunecontrol = svmCtr)
radical = svmObj1$best.parameters

rdSvm = svm(Class~.,data = dSVM_T,scale=TRUE,kernel="radial",cost=radical[2],gamma=radical[1],probability = TRUE)
dSVM_Test=subset(data_tsne,Class==-1)[-1]
dTestSVM = as.data.frame(dSVM_Test)
svmPred = predict(rdSvm,newdata = dTestSVM,probability = TRUE)
svmPreds = attr(svmPred,"probabilities")
colnames(svmPreds) = c("class1","class2","class3","class4","class5","class6","class7","class8","class9")
write.csv(svmPreds,"C:/Users/DC20693/Desktop/submissionsvmTest.csv")

#rf:imp~150+1500 tree
#names(dTrainSVM)=gsub(" ", "", names(dTrainSVM))
dTrainRF = dTrainSVM[,c("Class",imp2)]
dTestRF = dTestSVM[,imp2]
set.seed(81)
KMForest = randomForest(Class~., data=dTrainRF,
                        importance = TRUE,
                        replace = TRUE,
                        proximity=TRUE,
                        oob.prox = TRUE,
                        nPerm = 3,
                        ntree=1500)
#TestSVM=as.data.frame(dTest)
#names(dTestSVM)=gsub(" ", "", names(dTestSVM))
pred = predict(KMForest, dTestRF,type="prob")
write.csv(pred,"C:/Users/DC20693/Desktop/submissionRF_3.csv")

#gbm
h2o.init()
dataFinal$Class=factor(dataFinal$Class)
train_water = as.h2o(subset(dataFinal,Class!=-1L,select = c("Class",imp3))) 
test_water = as.h2o(subset(dataFinal,Class==-1L,select=imp3))
response = 1
predictors=2:100
splits = h2o.splitFrame(train_water, 0.7, destination_frames = c("trainSplit","validSplit"), seed = 82)
trainSplit = splits[[1]]
validSplit = splits[[2]]

ntrees_opts = c(10000) ## early stopping will stop earlier
max_depth_opts = seq(2,15)
min_rows_opts = c(1,5,10,20,50,100)
learn_rate_opts = seq(0.001,0.2,length.out = 5)
sample_rate_opts =seq(0.3,1,0.05)
col_sample_rate_opts = seq(0.4,1,0.1)
col_sample_rate_per_tree_opts = seq(0.4,1,0.1)

hyper_params = list( ntrees = ntrees_opts,
                     max_depth = max_depth_opts,
                     min_rows = min_rows_opts,
                     learn_rate = learn_rate_opts,
                     sample_rate = sample_rate_opts,
                     col_sample_rate = col_sample_rate_opts,
                     col_sample_rate_per_tree = col_sample_rate_per_tree_opts)

search_criteria = list(strategy = "RandomDiscrete", 
                       max_runtime_secs = 600, 
                       max_models = 200, 
                       stopping_metric = "logloss", 
                       stopping_tolerance = 0.00001, 
                       stopping_rounds = 5, seed = 82)

gbm.grid = h2o.grid(algorithm = "gbm",
                     grid_id="depth_grid",
                     x = predictors,
                     y = response,
                     training_frame = trainSplit, # alternatively, use N-fold cross-validation
                     validation_frame = validSplit,#training_frame = train,
                     nfolds = 0, #nfolds = 5,
                     distribution="multinomial", 
                     stopping_rounds = 10,
                     stopping_tolerance = 1e-3,
                     stopping_metric = "logloss",
                     score_tree_interval = 100, ## how often to score (affects early stopping)
                     seed = 82, ## seed to control the sampling of the Cartesian hyper-parameter space
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)
gbm.sorted.grid = h2o.getGrid(grid_id = "depth_grid", sort_by = "logloss")
best_model = h2o.getModel(gbm.sorted.grid@model_ids[[1]])

pred = h2o.predict(best_model,newdata = test_water,type="prob")
pred=as.matrix(pred)

h2o.shutdown(prompt = TRUE)
###################################
#vaildation                       #
###################################
#train-multiclass logloss
act = as.matrix(model.matrix(~Cheat-1,data=cheat))

#cheat
cheat = read.csv("C:/Users/DC20693/Desktop/CHEATRESUTL.csv")
cheatList = cheat$ID
preds = read.csv("C:/Users/DC20693/Desktop/submission912.csv")
predsCheat = as.matrix(preds[cheatList+1,-1])
predsCheat = predsCheat/rowSums(predsCheat)
predsCheat = ifelse(predsCheat == 0, 10**-15,predsCheat)
predsCheat = ifelse(predsCheat == 1, 1-10**-15,predsCheat)
cheat$Cheat=factor(cheat$Cheat)
act = as.matrix(model.matrix(~Cheat-1,data=cheat))
multiLogloss(act,predsCheat)
var(rowMeans(act*log(predsCheat)+(1-act)*log(1-predsCheat)))
