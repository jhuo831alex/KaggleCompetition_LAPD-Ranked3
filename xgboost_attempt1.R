
###############
dmy <- dummyVars("~.", data=full)
trainTrsf <- data.frame(predict(dmy, newdata = full))
outcomeName <- c('elapsed_time')
predictors <- names(trainTrsf)[!names(trainTrsf) %in% outcomeName]
trainPortion <- nrow(trainTrsf)
trainSet <- trainTrsf[ 1:floor(trainPortion/2),]
testSet <- trainTrsf[(floor(trainPortion/2)+1):trainPortion,]

cv <- 5
trainSet <- trainTrsf[1:trainPortion,]
cvDivider <- floor(nrow(trainSet) / (cv+1))
rounds<-100
depth<-5


smallestError <- 10000
#for (depth in 4:6) { 
  for (ETA in seq(5,30,5)) {
    for(Ga in 1:10){
    totalError <- c()
    indexCount <- 1
    for (cv in seq(1:cv)) {
      # assign chunk to data test
      dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
      dataTest <- trainSet[dataTestIndex,]
      # everything else to train
      dataTrain <- trainSet[-dataTestIndex,]
      
      bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                     label = dataTrain[,outcomeName],
                     max.depth=depth, 
                     nround=rounds,
                     eta=0.01*ETA,
                     gamma=Ga,
                     objective = "reg:linear", verbose=0)
      gc()
      predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
      
      err <- rmse(as.numeric(dataTest[,outcomeName]), as.numeric(predictions))
      totalError <- c(totalError, err)
    }
    if (mean(totalError) < smallestError) {
      smallestError = mean(totalError)
      print(paste(ETA,Ga,smallestError))
    }  
    }
  }
#} 

bst <- xgboost(data = as.matrix(trainSet[,predictors]),
               label = trainSet[,outcomeName],
               max.depth=depth, 
               nround=rounds, 
               objective = "reg:linear", 
               eta=ETA,
               gamma=Ga,
               verbose=0)
pred <- predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE)
rmse(as.numeric(testSet[,outcomeName]), as.numeric(pred))
