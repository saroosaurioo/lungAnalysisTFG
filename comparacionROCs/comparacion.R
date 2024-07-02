#LIBRERIA + PATH
library(pROC)
#Cambiar el directorio de trabajo
#setwd("C:/Users/sarit/OneDrive/Escritorio")

rocT <- read.csv("data_transfer.csv",sep=";")
rocF <- read.csv("data_fine.csv",sep=";")
rocB <- read.csv("data_base.csv", sep = ";")

rocT <- roc(rocT$y_test,rocT$predictions)
rocF <- roc(rocF$y_test,rocF$predictions)
rocB <- roc(rocB$y_test, rocB$predictions)


rocTsmooth <- roc(rocT$y_test,rocT$predictions, percent = FALSE, na.rm=TRUE, direction=c("auto", "<",">"), smooth=TRUE,auc=TRUE,ci=TRUE,plot=TRUE,smooth.method="binormal",density=NULL)
rocFsmooth <- roc(rocF$y_test,rocF$predictions, percent = FALSE, na.rm=TRUE, direction=c("auto", "<",">"), smooth=TRUE,auc=TRUE,ci=TRUE,plot=TRUE,smooth.method="binormal",density=NULL)
rocBsmooth <- roc(rocB$y_test,rocB$predictions, percent = FALSE, na.rm=TRUE, direction=c("auto", "<",">"), smooth=TRUE,auc=TRUE,ci=TRUE,plot=TRUE,smooth.method="binormal",density=NULL)
rocTsmooth
auc(rocT)
auc(rocF)
auc(rocBsmooth)


#ROC normalizadas conjuntas
#limpiar el dispositivo gráfico
dev.off()
plot.roc(rocTsmooth, col = "blue", main = "Curvas ROC", lwd = 2)
plot.roc(rocFsmooth, col = "red", add = TRUE)
plot.roc(rocBsmooth, col = "green", add = TRUE)
print(plot())


#COMPARACIÓN
roc.test(rocTsmooth,rocFsmooth,reuse.auc=FALSE)
roc.test(rocT,rocF,reuse.auc=FALSE,method="bootstrap")
roc.test(rocTsmooth,rocBsmooth,reuse.auc=FALSE)
roc.test(rocT,rocB,reuse.auc,method="bootstrap")
roc.test(rocT,rocF,reuse.auc=FALSE,method="delong")
roc.test(rocT,rocB,reuse.auc=FALSE,method="delong")

CT <- ci(rocT)

sens.ci <- ci.se(rocT, specificities=seq(0, 100, 5))
plot(sens.ci, type="shape", col="lightblue")
plot(sens.ci, type="bars")
ci(rocFsmooth)


#PARCIALES

roc1 <- roc(rocT$y_test,
            rocT$prediction, percent=TRUE,
            # arguments for auc
            partial.auc=c(100, 90), partial.auc.correct=TRUE,
            partial.auc.focus="sens",
            # arguments for ci
            ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)



roc2 <- roc(rocF$y_test,
            rocF$prediction, percent=TRUE,
            # arguments for auc
            partial.auc=c(100, 90), partial.auc.correct=TRUE,
            partial.auc.focus="sens",
            # arguments for ci
            ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)

#COMPARACIÓN
roc.test(roc1,roc2,reuse.auc=FALSE,method="delong")
roc.test(roc1,roc2,reuse.auc=FALSE,method="bootstrap")



rocS1 <- roc(rocT$y_test,
            rocT$prediction, percent=TRUE,
            # arguments for auc
            partial.auc=c(100, 90), partial.auc.correct=TRUE,
            # arguments for ci
            ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)



rocS2 <- roc(rocF$y_test,
            rocF$prediction, percent=TRUE,
            # arguments for auc
            partial.auc=c(100, 90), partial.auc.correct=TRUE,
            # arguments for ci
            ci=TRUE, boot.n=100, ci.alpha=0.9, stratified=FALSE,
            # arguments for plot
            plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)

roc.test(rocS1,rocS2,reuse.auc=FALSE,method="delong")
roc.test(rocS1,rocS2,reuse.auc=FALSE,method="bootstrap")



