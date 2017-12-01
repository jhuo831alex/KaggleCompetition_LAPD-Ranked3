library(xgboost)
library(caret)
library(Metrics)
library(dplyr)
library(mlr)
library(Hmisc)
library(checkmate)

full <- read.csv("D:/Dropbox/UCLA/[Courses]/Stats Minor/STATS 101C/Project 101C/lafdtraining updated.csv")
full$fd <- substring(full$incident.ID,1,7)
full<-full[which(!is.na(full$elapsed_time)),]
full$incident <- substring(full$incident.ID,9,length(full$incident.ID))
full$incident<-as.numeric(full$incident)
c <- full %>% group_by(incident.ID) %>% summarise(cnt=n())
full<-merge(full,c,by.x="incident.ID")
full$Incident.Creation.Time..GMT.<-as.numeric(full$Incident.Creation.Time..GMT.)
full$creation <- cut(full$Incident.Creation.Time..GMT., 
                     breaks = c(0, 6*60*60, 12*60*60, 18*60*60,max(full$Incident.Creation.Time..GMT.)), 
                     labels = c(1,2,3,4))
full$incident.ID<-NULL
full$row.id<-NULL
full$Emergency.Dispatch.Code<-NULL
full$year<-as.factor(full$year)
full$First.in.District<-as.factor(full$First.in.District)
full$fd<-as.factor(full$fd)
full$Incident.Creation.Time..GMT.<-NULL

table(full$First.in.District)
