# Universidad del Valle de Guatemala
# Mineria de Datos - Seccion 10
# Integrantes: Oscar Juarez, Jose Cifuentes, Luis Esturban
# Fecha: 25/04/20

#HOJA DE TRABAJO 7: REDES NEURONALES

#Importar librerias
library(caret)
library(nnet)
library(RWeka)
library(neural)
library(dummy)
library(neuralnet)
library(e1071)

# Leer datos del csv
data <- read.csv("./Data/train.csv", stringsAsFactors = FALSE)
testing <- read.csv("./Data/test.csv", stringsAsFactors = FALSE)

# Se hace una categoria para el precio de cada casa.
grupoRespuesta <- c()
for (value in data[,"SalePrice"]) {
  if (value <= 260400) {
    grupoRespuesta <- c(grupoRespuesta, 1)
  } else if (value >= 410000) {
    grupoRespuesta <- c(grupoRespuesta, 2) 
  } else {
    grupoRespuesta <- c(grupoRespuesta, 3)
  }
}
data$grupoRespuesta <- grupoRespuesta

# Datos a utilizar
set.seed(69)
porcentaje<-0.7
corte <- sample(nrow(data),nrow(data)*porcentaje)
train<-data[corte,]
test<-data[-corte,]

varNames <- c("MSSubClass","LotArea","OverallQual","OverallCond","X1stFlrSF","BsmtFullBath","BedroomAbvGr","GarageCars","grupoRespuesta")
TestvarNames <- c("MSSubClass","LotArea","OverallQual","OverallCond","X1stFlrSF","BsmtFullBath","BedroomAbvGr","GarageCars")







#-------------------------------------------------
# Red Neuronal con NeuralNet (logistic)
#-------------------------------------------------
start_time <- Sys.time()
train$y<-as.numeric(train$grupoRespuesta)
test$y<-as.numeric(test$grupoRespuesta)
#'logistic' and 'tanh' are possible for the logistic function and tangent hyperbolicus.
modelo.nn <- neuralnet(grupoRespuesta~., train[,varNames], hidden = 9, rep = 1,act.fct = "logistic",linear.output = FALSE)
#plot(modelo.nn, newdata=test) #Sale un gr?fico por cada repetici?n, en este caso saldr?n 3 gr?ficos
test$predNeuralNet<-round(predict(modelo.nn,newdata = test[,TestvarNames]),0)
cfmNeuralNet<-confusionMatrix(as.factor(test$predNeuralNet),as.factor(test$y))
cfmNeuralNet
end_time <- Sys.time()
end_time - start_time

#-------------------------------------------------
# Red Neuronal con NeuralNet (tangent hyperbolicus)
#-------------------------------------------------
start_time <- Sys.time()
train$y<-as.numeric(train$grupoRespuesta)
test$y<-as.numeric(test$grupoRespuesta)
#'logistic' and 'tanh' are possible for the logistic function and tangent hyperbolicus.
modelo.nn <- neuralnet(grupoRespuesta~., train[,varNames], hidden = c(3,3,3), rep = 1,act.fct = "tanh",linear.output = FALSE)
#plot(modelo.nn, newdata=test) #Sale un gr?fico por cada repetici?n, en este caso saldr?n 3 gr?ficos
test$predNeuralNet<-round(predict(modelo.nn,newdata = test[,TestvarNames]),0)
cfmNeuralNet<-confusionMatrix(as.factor(test$predNeuralNet),as.factor(test$y))
cfmNeuralNet
end_time <- Sys.time()
end_time - start_time
#-------------------------------
#SVM: Polinomial 1
#Grado:9
#kernel:Polynomial
#-------------------------------
modeloSVM_L<-svm(grupoRespuesta~., data=train[,varNames], degree=9, kernel="polynomial") #98%
prediccionL<-predict(modeloSVM_L,newdata=test[,TestvarNames])
testcompleto<-test[,TestvarNames]
testcompleto$pred<-trunc(prediccionL)
testcompleto$pred1<-ifelse(testcompleto$pred <= 1, 1,ifelse( testcompleto$pred >=1.1 & testcompleto$pred <= 2.5 , 2 ,ifelse( testcompleto$pred >=2.6, 3,"hola"  ) ) )
str(as.factor(testcompleto$pred1))
str(as.factor(test$grupoRespuesta))
confusionMatrix(as.factor(testcompleto$pred1),as.factor(test$grupoRespuesta))
#-------------------------------
#SVM: Polinomial 2
#Grado:4
#kernel:Polynomial
#-------------------------------
modeloSVM_L<-svm(grupoRespuesta~., data=train[,varNames], degree=4, kernel="polynomial") #98%
prediccionL<-predict(modeloSVM_L,newdata=test[,TestvarNames])
testcompleto<-test[,TestvarNames]
testcompleto$pred<-trunc(prediccionL)
testcompleto$pred1<-ifelse(testcompleto$pred <= 1, 1,ifelse( testcompleto$pred >=1.1 & testcompleto$pred <= 2.5 , 2 ,ifelse( testcompleto$pred >=2.6, 3,"hola"  ) ) )
str(as.factor(testcompleto$pred1))
str(as.factor(test$grupoRespuesta))
confusionMatrix(as.factor(testcompleto$pred1),as.factor(test$grupoRespuesta))
#-------------------------------
#SVM: Radial 1
#Gamma:3^6
#kernel:Radial
#-------------------------------
modeloSVM_L<-svm(grupoRespuesta~., data=train[,varNames], gamma=3^6, kernel="radial") #98%
prediccionL<-predict(modeloSVM_L,newdata=test[,TestvarNames])
testcompleto<-test[,TestvarNames]
testcompleto$pred<-trunc(prediccionL)
testcompleto$pred2<-ifelse(testcompleto$pred <= 1, 1,ifelse( testcompleto$pred >=1.1 & testcompleto$pred <= 2.5 , 2 ,ifelse( testcompleto$pred >=2.6, 3,"hola"  ) ) )
str(as.factor(testcompleto$pred2))
str(as.factor(test$grupoRespuesta))
confusionMatrix(as.factor(testcompleto$pred2),as.factor(test$grupoRespuesta))
#-------------------------------
#SVM: Radial 2
#Gamma:5^6
#kernel:Radial
#-------------------------------
modeloSVM_L<-svm(grupoRespuesta~., data=train[,varNames], gamma=5^6, kernel="radial") #98%
prediccionL<-predict(modeloSVM_L,newdata=test[,TestvarNames])
testcompleto<-test[,TestvarNames]
testcompleto$pred<-trunc(prediccionL)
testcompleto$pred2<-ifelse(testcompleto$pred <= 1, 1,ifelse( testcompleto$pred >=1.1 & testcompleto$pred <= 2.5 , 2 ,ifelse( testcompleto$pred >=2.6, 3,"hola"  ) ) )
str(as.factor(testcompleto$pred2))
str(as.factor(test$grupoRespuesta))
confusionMatrix(as.factor(testcompleto$pred2),as.factor(test$grupoRespuesta))
#-------------------------------
#SVM: Linear 1
#Costo:10
#Gamma:3
#kernel:Linear
#-------------------------------
modeloSVM_L<-svm(grupoRespuesta~., data=train[,varNames], cost=10, gamma=3, kernel="linear")
prediccionL<-predict(modeloSVM_L,newdata=test[,TestvarNames])
testcompleto<-test[,TestvarNames]
testcompleto$pred<-ceiling(prediccionL)
testcompleto$pred2<-ifelse(testcompleto$pred <= 1, 1,ifelse( testcompleto$pred >=1.1 & testcompleto$pred <= 2.5 , 2 ,ifelse( testcompleto$pred >=2.6, 3,"hola"  ) ) )
str(as.factor(testcompleto$pred2))
str(as.factor(test$grupoRespuesta))
confusionMatrix(as.factor(testcompleto$pred2),as.factor(test$grupoRespuesta))
#-------------------------------
#SVM: Linear 2
#Costo:17
#Gamma:6
#kernel:Linear
#-------------------------------
modeloSVM_L<-svm(grupoRespuesta~., data=train[,varNames], cost=17, gamma=6, kernel="linear")
prediccionL<-predict(modeloSVM_L,newdata=test[,TestvarNames])
testcompleto<-test[,TestvarNames]
testcompleto$pred<-ceiling(prediccionL)
testcompleto$pred2<-ifelse(testcompleto$pred <= 1, 1,ifelse( testcompleto$pred >=1.1 & testcompleto$pred <= 2.5 , 2 ,ifelse( testcompleto$pred >=2.6, 3,"hola"  ) ) )
str(as.factor(testcompleto$pred2))
str(as.factor(test$grupoRespuesta))
confusionMatrix(as.factor(testcompleto$pred2),as.factor(test$grupoRespuesta))
