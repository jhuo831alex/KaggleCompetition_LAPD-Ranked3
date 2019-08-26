library(xgboost)
library(caret)
library(Metrics)
library(dplyr)
library(mlr)
library(Hmisc)
library(checkmate)

makeRLearner.regr.xgboost = function() {
  makeRLearnerRegr(
    cl = "regr.xgboost",
    package = "xgboost",
    par.set = makeParamSet(
      # we pass all of what goes in 'params' directly to ... of xgboost
      #makeUntypedLearnerParam(id = "params", default = list()),
      makeDiscreteLearnerParam(id = "booster", default = "gbtree", values = c("gbtree", "gblinear")),
      makeIntegerLearnerParam(id = "silent", default = 0),
      makeNumericLearnerParam(id = "eta", default = 0.3, lower = 0),
      makeNumericLearnerParam(id = "gamma", default = 0, lower = 0),
      makeIntegerLearnerParam(id = "max_depth", default = 6, lower = 0),
      makeNumericLearnerParam(id = "min_child_weight", default = 1, lower = 0),
      makeNumericLearnerParam(id = "subsample", default = 1, lower = 0, upper = 1),
      makeNumericLearnerParam(id = "colsample_bytree", default = 1, lower = 0, upper = 1),
      makeIntegerLearnerParam(id = "num_parallel_tree", default = 1, lower = 1),
      makeNumericLearnerParam(id = "lambda", default = 0, lower = 0),
      makeNumericLearnerParam(id = "lambda_bias", default = 0, lower = 0),
      makeNumericLearnerParam(id = "alpha", default = 0, lower = 0),
      makeUntypedLearnerParam(id = "objective", default = "reg:linear"),
      makeUntypedLearnerParam(id = "eval_metric", default = "rmse"),
      makeNumericLearnerParam(id = "base_score", default = 0.5),
      
      makeNumericLearnerParam(id = "missing", default = 0),
      makeIntegerLearnerParam(id = "nthread", default = 16, lower = 1),
      makeIntegerLearnerParam(id = "nrounds", default = 1, lower = 1),
      makeUntypedLearnerParam(id = "feval", default = NULL),
      makeIntegerLearnerParam(id = "verbose", default = 1, lower = 0, upper = 2),
      makeIntegerLearnerParam(id = "print_every_n", default = 1, lower = 1),
      makeIntegerLearnerParam(id = "early_stopping_rounds", default = 1, lower = 1),
      makeLogicalLearnerParam(id = "maximize", default = FALSE)
    ),
    par.vals = list(nrounds = 1),
    properties = c("numerics", "factors", "weights"),
    name = "eXtreme Gradient Boosting",
    short.name = "xgboost",
    note = "All settings are passed directly, rather than through `xgboost`'s `params` argument. `nrounds` has been set to `1` by default."
  )
}
trainLearner.regr.xgboost = function(.learner, .task, .subset, .weights = NULL,  ...) {
  td = getTaskDescription(.task)
  data = getTaskData(.task, .subset, target.extra = TRUE)
  target = data$target
  data = data.matrix(data$data)
  
  parlist = list(...)
  obj = parlist$objective
  if (checkmate::testNull(obj)) {
    obj = "reg:linear"
  }
  
  if (checkmate::testNull(.weights)) {
    xgboost::xgboost(data = data, label = target, objective = obj, ...)
  } else {
    xgb.dmat = xgboost::xgb.DMatrix(data = data, label = target, weight = .weights)
    xgboost::xgboost(data = xgb.dmat, label = NULL, objective = obj, ...)
  }
}
predictLearner.regr.xgboost = function(.learner, .model, .newdata, ...) {
  td = .model$task.desc
  m = .model$learner.model
  xgboost::predict(m, newdata = data.matrix(.newdata), ...)
}
trainTask = makeRegrTask(data = full, target = "elapsed_time")
trainTask = createDummyFeatures(trainTask)
#testTask = makeRegrTask(data = trytest, target = "elapsed_time")
#testTask = createDummyFeatures(testTask)

lrn = makeLearner("regr.xgboost")
lrn$par.vals = list(
  nthread             = 5,
  nrounds             = 100,
  print_every_n       = 50,
  objective           = "reg:linear"
)
# missing values will be imputed by their median
lrn = makeImputeWrapper(lrn, classes = list(numeric = imputeMedian(), integer = imputeMedian()))

SQWKfun = function(x = seq(1.5, 7.5, by = 1), pred) {
  preds = pred$data$response
  true = pred$data$truth 
  cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
  preds = as.numeric(Hmisc::cut2(preds, cuts))
  err = Metrics::ScoreQuadraticWeightedKappa(preds, true, 1, 8)
  return(-err)
}

SQWK = makeMeasure(id = "SQWK", minimize = FALSE, properties = c("regr"), best = 1, worst = 0,
                   fun = function(task, model, pred, feats, extra.args) {
                     return(-SQWKfun(x = seq(1.5, 7.5, by = 1), pred))
                   })

# Do it in parallel with parallelMap
library(parallelMap)
parallelStartSocket(3)
parallelExport("SQWK", "SQWKfun", "trainLearner.regr.xgboost", "predictLearner.regr.xgboost" , "makeRLearner.regr.xgboost")
## This is how you could do hyperparameter tuning
# # 1) Define the set of parameters you want to tune (here 'eta')
ps = makeParamSet(
  makeNumericParam("eta", lower = 0.1, upper = 0.3),
  makeNumericParam("colsample_bytree", lower = 1, upper = 2, trafo = function(x) x/2),
  makeNumericParam("subsample", lower = 1, upper = 2, trafo = function(x) x/2)
)
getParamSet(lrn)
# # 2) Use 10-fold Cross-Validation to measure improvements
rdesc = makeResampleDesc("CV", iters = 10L)
# # 3) Here we use Random Search (with 10 Iterations) to find the optimal hyperparameter
ctrl =  makeTuneControlRandom(maxit = 10)
# # 4) now use the learner on the training Task with the 3-fold CV to optimize your set of parameters and evaluate it with SQWK
res = tuneParams(lrn, task = trainTask, resampling = rdesc, par.set = ps, control = ctrl, measures = SQWK)
res

#######################################
dmy <- dummyVars("~.", data=full)
trainTrsf <- data.frame(predict(dmy, newdata = full))
outcomeName <- c('elapsed_time')
predictors <- names(trainTrsf)[!names(trainTrsf) %in% outcomeName]
trainSet <- trainTrsf
bst <- xgboost(data = as.matrix(trainSet[,predictors]),
               label = trainSet[,outcomeName],
               objective = "reg:linear", 
               nrounds=100,
               verbose=0,
               eta=0.995,
               colsample_bytree=0.502,
               subsample=0.973
)
##############################################
testing <- read.csv("~/Desktop/testing.csv")
testing$elapsed_time<-0
testing$fd <- substring(testing$incident.ID,1,7)
testing<-testing[which(!is.na(testing$elapsed_time)),]
testing$incident <- substring(testing$incident.ID,9,length(testing$incident.ID))
testing$incident<-as.numeric(testing$incident)
c <- testing %>% group_by(incident.ID) %>% summarise(cnt=n())
testing<-merge(testing,c,by.x="incident.ID")
testing$Incident.Creation.Time..GMT.<-as.numeric(testing$Incident.Creation.Time..GMT.)
testing$creation <- cut(testing$Incident.Creation.Time..GMT., 
                     breaks = c(0, 1*60*60, 2*60*60, 3*60*60, 4*60*60,
                                5*60*60,6*60*60,7*60*60,8*60*60,
                                9*60*60,10*60*60,11*60*60,12*60*60, 13*60*60, 14*60*60,
                                15*60*60,16*60*60,17*60*60,18*60*60,
                                19*60*60,20*60*60,21*60*60,22*60*60,23*60*60,
                                max(testing$Incident.Creation.Time..GMT.)), 
                     labels = c(1:24))
testing$incident.ID<-NULL
testing$row.id<-NULL
testing$Emergency.Dispatch.Code<-NULL
testing$year<-as.factor(testing$year)
testing$First.in.District<-as.factor(testing$First.in.District)
testing$fd<-as.factor(testing$fd)

fulltest<-rbind(full,testing)

dmytest <- dummyVars("~.", data=fulltest)
testTrsf <- data.frame(predict(dmytest, newdata = fulltest))
pred <- predict(bst, as.matrix(testTrsf[,predictors]), outputmargin=TRUE)
# rmse(as.numeric(testSet[,outcomeName]), as.numeric(pred))
fultest$pred<-pred
result<-fultest[fultest$elapsed_time==0,]
final_result<-data.frame(row.id=testing$row.id,prediction=result$pred)
write.csv(final_result,file="attempt1.csv",row.names=F)
