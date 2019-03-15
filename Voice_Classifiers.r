setwd("C:\\Users\\user\\Desktop\\Jigsaw\\Introduction to Machine Learning\\Codes of Machine Learning")
voice=read.csv("voice.csv")
head(voice)
dim(voice)
colSums(is.na(voice))
table(voice$label)

library(ROCR)
library(caret)
library(car)
library(DiscriMiner)

trainIndex=createDataPartition(voice$label, p=0.75, list = FALSE,times = 1)
voice.train=voice[trainIndex,]
voice.test=voice[-trainIndex,]
voice.train.x=voice[trainIndex,1:20]
voice.train.y=voice[trainIndex,21]

voice.test.x=voice[-trainIndex,1:20]
voice.test.y=voice[-trainIndex,21]


###Random Forest

library(randomForest)
library(Boruta)

voice.BT=Boruta(label~., data = voice, doTrace=2, ntree=500)
plot(voice.BT, xlab = "", xaxt = "n", main="Variable Importance")
k <-lapply(1:ncol(voice.BT$ImpHistory),function(i)
  voice.BT$ImpHistory[is.finite(voice.BT$ImpHistory[,i]),i])
names(k) <- colnames(voice.BT$ImpHistory)
Labels <- sort(sapply(k,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(voice.BT$ImpHistory), cex.axis = 0.7)


getSelectedAttributes(voice.BT)
attStats(voice.BT)

set.seed(1234)
ctrl=trainControl(method = "repeatedcv", number = 10, repeats = 1)
set.seed(1234)

#Grid Search to obtain the best parameters to build a random forest model

tunegrid <- expand.grid(.mtry=seq(2,18,1))
rf_gridsearch <- train(label~., data= voice.train, method="rf",
                       tuneGrid=tunegrid, trControl=ctrl)
print(rf_gridsearch)
plot(rf_gridsearch)

#best value of mtry was 6

voice.RF=randomForest(label~., data = voice.train,mtry= 6, importance=TRUE)
plot(voice.RF, main="")
legend("topright", c("OOB", "0", "1"), text.col=1:6, lty=1:3, col=1:3)
title(main="Error Rates Random Forest-Final")


impVar <- round(randomForest::importance(voice.RF), 2)
impVar[order(impVar[,3], decreasing=TRUE),]

voice.pred.class=predict(voice.RF, voice.test, type = "class")
CM=confusionMatrix(voice.test[,21],voice.pred.class)
fourfoldplot(CM$table)
Accuracy_RF=CM$overall[1]
Sensitivity_RF=CM$byClass[1]
Specificity_RF=CM$byClass[2]

voice.prediction=prediction(as.numeric(voice.pred.class),voice.test$label)
voice.perf=performance(voice.prediction,"tpr","fpr")
plot(voice.perf)
auc=performance(voice.prediction,measure = "auc")
AUC_RF=auc@y.values[[1]]


##################
#Build a SVM model using Linear kernel
###Tuning parameter C for optimized model
grid=expand.grid(C = c(0.01, 0.02,0.05, 0.075, 0.1, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2,5))

set.seed(1234)
voice.SVM_Lin=train(voice.train.x,voice.train.y,
                    tuneGrid=grid, tuneControl=ctrl,
                    preProcess=c("scale","center"),
                    method='svmLinear')
voice.SVM_Lin$bestTune
plot(voice.SVM_Lin)

voice.pred=predict(voice.SVM_Lin, voice.test.x)

voice.prediction=prediction(as.numeric(voice.pred),voice.test$label)
voice.perf=performance(voice.prediction,"tpr","fpr")
plot(voice.perf)
auc=performance(voice.prediction,measure = "auc")
AUC_SVM_Lin=auc@y.values[[1]]

CM=confusionMatrix(voice.test[,21],voice.pred)
fourfoldplot(CM$table)
Accuracy_SVM_Lin=CM$overall[1]
Sensitivity_SVM_Lin=CM$byClass[1]
Specificity_SVM_Lin=CM$byClass[2]

#######
#Build a SVM model using polynomial kernel
###Tuning parameter C for optimized model
grid=expand.grid(C = c(0.01, 0.02,0.05, 0.075, 0.1, 
                       0.25, 0.5, 1, 1.25, 1.5, 1.75, 2,5),
                 degree=c(2,3), scale=1)

set.seed(12345)
voice.SVM_Pol=train(voice.train.x,voice.train.y,
                    tuneGrid=grid, tuneControl=ctrl,
                                      method='svmPoly')
voice.SVM_Pol$bestTune
plot(voice.SVM_Pol)

voice.pred=predict(voice.SVM_Pol, voice.test.x)

voice.prediction=prediction(as.numeric(voice.pred),voice.test$label)
voice.perf=performance(voice.prediction,"tpr","fpr")
plot(voice.perf)
auc=performance(voice.prediction,measure = "auc")
AUC_SVM_Pol=auc@y.values[[1]]

CM=confusionMatrix(voice.test[,21],voice.pred)
fourfoldplot(CM$table)
Accuracy_SVM_Pol=CM$overall[1]
Sensitivity_SVM_Pol=CM$byClass[1]
Specificity_SVM_Pol=CM$byClass[2]

#######
#Build a SVM model using Radial kernel
###Tuning parameter C for optimized model
grid=expand.grid(C = c(.01, 0.1, 0.20,0.5), 
                 sigma=c(0.005,0.01,0.02,0.025))

set.seed(123465)
voice.SVM_Rad=train(voice.train.x,voice.train.y,
                    tuneGrid=grid, tuneControl=ctrl,
                    method='svmRadial')
voice.SVM_Rad$bestTune
plot(voice.SVM_Rad)

voice.pred=predict(voice.SVM_Rad, voice.test.x)

voice.prediction=prediction(as.numeric(voice.pred),voice.test$label)
voice.perf=performance(voice.prediction,"tpr","fpr")
plot(voice.perf)
auc=performance(voice.prediction,measure = "auc")
AUC_SVM_Rad=auc@y.values[[1]]

CM=confusionMatrix(voice.test[,21],voice.pred)
fourfoldplot(CM$table)
Accuracy_SVM_Rad=CM$overall[1]
Sensitivity_SVM_Rad=CM$byClass[1]
Specificity_SVM_Rad=CM$byClass[2]

###############################################
#Building a Logistic Regression Model
str(voice.train)
voice.train$label=factor(ifelse(voice.train$label=="male",1,0))
library(corrplot)
M=cor(na.omit(voice.train[,-21]))
corrplot(M, method = "circle", type = "lower", 
         tl.srt = 45, tl.col = "black", tl.cex = 0.75)


#meanfreq and centroid are identical with a correlation of 1. centroid can be removed.
#median,Q25 are also highly correlated with meanfreq.
#maxdom and dfrange are also highly correlated. One of them to be removed
#centroid,median,dfrange and Q25 could be be removed to reduce multicollienearity

voice.train.Cleaned=voice.train[,-c(3,4,12,19)]
voice.test.Cleaned=voice.test[,-c(3,4,12,19)]

#Build model and check for multicolinearity using VIF
LogMod1=glm(label~.,data = voice.train.Cleaned,family = 'binomial')
summary(LogMod1)
options(scipen = 9999)
car::vif(LogMod1)

## Variables with very high VIF (greater than 10) could
#be removed to eliminate multicollinearity.
#meanfreq,sd,Q75,,IQR,skew,kurt,sfm could be removed while building the model

names(voice.train.Cleaned)
voice.train.Cleaned=voice.train.Cleaned[,-c(1:6,8)]
voice.test.Cleaned=voice.test.Cleaned[,-c(1:6,8)]


LogMod2=glm(label~.,data = voice.train.Cleaned,family = 'binomial')
summary(LogMod2)
options(scipen = 9999)
car::vif(LogMod2)
#VIF is now stable.
#Check for the best model by removing variables that are not significant
step(LogMod2, direction = "both")

LogModF=glm(formula = label ~ sp.ent + mode + meanfun + minfun + meandom + 
              mindom + modindx, family = "binomial", data = voice.train.Cleaned)
summary(LogModF)

###Prediction
voice.pred=predict(LogModF, voice.test.Cleaned, type = "response")
voice.class=ifelse(voice.pred>0.5,"male","female")

voice.prediction=prediction(as.numeric(voice.pred),voice.test.Cleaned$label)
voice.perf=performance(voice.prediction,"tpr","fpr")
plot(voice.perf)
auc=performance(voice.prediction,measure = "auc")
AUC_Log=auc@y.values[[1]]

CM=confusionMatrix(voice.test.Cleaned$label,as.factor(voice.class))
fourfoldplot(CM$table)
Accuracy_Log=CM$overall[1]
Sensitivity_Log=CM$byClass[1]
Specificity_Log=CM$byClass[2]


####################################
#Building Linear Discriminant Model
#Since Multicollinearity impacts Linear Discriminant Model like the Logistic model
#the same cleaned data was used as incase of Logistic Model

library(MASS)
library(DiscriMiner)
DiscM=lda(label~.,data = voice.train.Cleaned)
names(voice.train.Cleaned)
M=manova(as.matrix(voice.train.Cleaned[,c(1:9)])
         ~as.matrix(voice.train.Cleaned[,10]))
summary(M)
summary.aov(M)

###Prediction
voice.pred=predict(DiscM, voice.test.Cleaned)
voice.prediction=prediction(as.numeric(voice.pred$class),voice.test.Cleaned$label)
voice.perf=performance(voice.prediction,"tpr","fpr")
plot(voice.perf)
auc=performance(voice.prediction,measure = "auc")
AUC_LDA=auc@y.values[[1]]
levels(voice.test.Cleaned$label)=c(0,1)
CM=confusionMatrix(factor(voice.test.Cleaned$label),(voice.pred$class))
fourfoldplot(CM$table)
Accuracy_LDA=CM$overall[1]
Sensitivity_LDA=CM$byClass[1]
Specificity_LDA=CM$byClass[2]

########################################
Compare=data.frame(Classifier=c("RF","SVM_Lin","SV_Pol","SV_Radial","Logistic", "LDA"),
                   Acc=c(Accuracy_RF,Accuracy_SVM_Lin,Accuracy_SVM_Pol,Accuracy_SVM_Rad,Accuracy_Log, Accuracy_LDA),
                   Sensitivity=c(Sensitivity_RF,Sensitivity_SVM_Lin,Sensitivity_SVM_Pol,Sensitivity_SVM_Rad,Sensitivity_Log,Sensitivity_LDA),
                   Specificity=c(Specificity_RF,Specificity_SVM_Lin,Specificity_SVM_Pol,Specificity_SVM_Rad,Specificity_Log, Specificity_LDA),
                   AUC_All=c(AUC_RF,AUC_SVM_Lin,AUC_SVM_Pol,AUC_SVM_Rad,AUC_Log, AUC_LDA))
Compare


library(reshape)
ggplot(melt(Compare,id.vars = "Classifier"),aes(Classifier,value, col=variable, group=variable))+geom_line()+
  geom_point(size=4,shape=21,fill="white")+
  labs(x="",y="Values", title="Evaluation Metric Comparison", color="Metrics")+
  theme(legend.key = element_rect(colour = "black", fill = "light blue"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 15, hjust = 0.5))

ggplot(melt(Compare,id.vars = "Classifier"),aes(x=variable,value, fill=Classifier))+
  geom_bar(stat = "identity", position = "dodge")+coord_flip()+
  labs(x="",y="Values", title="Evaluation Metric Comparison", color="Metrics")+
  theme(legend.key = element_rect(colour = "black", fill = "light blue"),
        axis.text.y = element_text(size = 10, hjust = 1, face = "bold"),
        plot.title = element_text(size = 15, hjust = 0.5),
        legend.key.size = unit(0.5,"cm"),
        legend.position = "bottom",
        legend.background = element_rect(fill="grey"))
