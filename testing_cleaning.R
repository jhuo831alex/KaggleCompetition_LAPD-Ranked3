testing <- read.csv("~/Desktop/testing.csv")
testing$elapsed_time<-0
testing$fd <- substring(testing$incident.ID,1,7)

testing$incident <- substring(testing$incident.ID,9,length(testing$incident.ID))
testing[,c(3,12)]<-lapply(testing[,c(3,12)],factor)
testing$incident<-as.numeric(testing$incident)
testing$Incident.Creation.Time..GMT.<-as.numeric(testing$Incident.Creation.Time..GMT.)
testing$Emergency.Dispatch.Code <- NULL
testing<-testing[,-c(1,2)]
testing$creation <- cut(testing$Incident.Creation.Time..GMT., 
                        breaks = c(0, 6*60*60, 12*60*60, 18*60*60,max(testing$Incident.Creation.Time..GMT.)), 
                        labels = c(1,2,3,4))
testing$Incident.Creation.Time..GMT. <- NULL
dmytest <- dummyVars("~.", data=testing)
testTrsf <- data.frame(predict(dmytest, newdata = testing))

fultest<-rbind(full,testing)
write.csv(fultest,file="fulltest.csv")

dmytest <- dummyVars("~.", data=fultest)
testTrsf <- data.frame(predict(dmytest, newdata = fultest))
pred <- predict(bst, as.matrix(testTrsf[,predictors]), outputmargin=TRUE)
# rmse(as.numeric(testSet[,outcomeName]), as.numeric(pred))
fultest$pred<-pred
result<-fultest[fultest$elapsed_time==0,]
final_result<-data.frame(row.id=testing$row.id,prediction=result$pred)
write.csv(final_result,file="attempt1.csv",row.names=F)
