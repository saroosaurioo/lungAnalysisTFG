library(pROC)
#Cambiar el directorio de trabajo
#setwd("C:/Users/sarit/OneDrive/Escritorio")

roc <- read.csv("data_modelo3.csv",sep=";")

head(roc)

multiclass.roc(roc$y_val, roc$predictions)
