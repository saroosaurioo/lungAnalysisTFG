#LIBRERIA + PATH
library(pROC)
#Cambiar el directorio de trabajo
#setwd("C:/Users/sarit/OneDrive/Escritorio")

rocT <- read.csv("data_transfer.csv",sep=";")
rocB2 <- read.csv("data_base_vgg2.csv",sep=";")
rocD2 <- read.csv("data_dataaug_vgg2.csv",sep=";")
rocB3 <- read.csv("data_base_vgg3.csv",sep=";")
rocD3 <- read.csv("data_dataaug_vgg3.csv",sep=";")
rocB <- read.csv("data_base.csv", sep = ";")


rocT <- roc(rocT$y_test,rocT$predictions)
rocB2 <- roc(rocB2$y_test,rocB2$predictions)
rocD2 <- roc(rocD2$y_test,rocD2$predictions)
rocB3 <- roc(rocB3$y_test,rocB3$predictions)
rocD3 <- roc(rocD3$y_test,rocD3$predictions)
rocB <- roc(rocB$y_test,rocB$predictions)


auc(rocT)
auc(rocB2)
auc(rocD2)
auc(rocB3)
auc(rocD3)
auc(rocB)

#Comparación Transfer learning vs VGG2
roc.test(rocT,rocB2,reuse.auc=FALSE,method="bootstrap")
roc.test(rocT,rocB2,reuse.auc=FALSE,method="delong")

roc.test(rocT,rocD2,reuse.auc=FALSE,method="bootstrap")
roc.test(rocT,rocD2,reuse.auc=FALSE,method="delong")

#Comparación VGG2 base vs data augmentation
roc.test(rocB2,rocD2,reuse.auc=FALSE,method="bootstrap")
roc.test(rocB2,rocD2,reuse.auc=FALSE,method="delong")

#Comparación Transfer learning vs VGG3
roc.test(rocT,rocB3,reuse.auc=FALSE,method="bootstrap")
roc.test(rocT,rocB3,reuse.auc=FALSE,method="delong")

roc.test(rocT,rocD3,reuse.auc=FALSE,method="bootstrap")
roc.test(rocT,rocD3,reuse.auc=FALSE,method="delong")

#Comparación base VGG16 vs data augmentation VGG3
roc.test(rocB,rocD3,reuse.auc=FALSE,method="bootstrap")
roc.test(rocB,rocD3,reuse.auc=FALSE,method="delong")


#Comparación VGG2 vs VGG3
roc.test(rocB2,rocB3,reuse.auc=FALSE,method="bootstrap")
roc.test(rocB2,rocB3,reuse.auc=FALSE,method="delong")

roc.test(rocD2,rocD3,reuse.auc=FALSE,method="bootstrap")
roc.test(rocD2,rocD3,reuse.auc=FALSE,method="delong")


#Comparación VGG3 base vs data augmentation
roc.test(rocB3,rocD3,reuse.auc=FALSE,method="bootstrap")
roc.test(rocB3,rocD3,reuse.auc=FALSE,method="delong")





