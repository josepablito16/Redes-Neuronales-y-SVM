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


