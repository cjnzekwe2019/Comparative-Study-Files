###########################################################################

rm(list = ls())

library(tidyverse)
library(MASS)        # for mvrnorm
library(mvnfast)     # for dmvn
library(Matrix)  
library(caret)
library(glmnet)
library(ncvreg)
library(RAMP)
library(iRF)          # For this study we used iRF 2.0.0 (`devtools::install_github("karlkumbier/iRF2.0")`)
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(Boruta)       # Selection of important features for RF
library(parallel)

(core.num <- detectCores())

setwd("Path")

Dat.cancer <- read.csv(Dat.URL, header = FALSE, sep = ",", quote = "\"'")
names(Dat.cancer) <- c('id_number', 'diagnosis', 'radius_mean', 
                       'texture_mean', 'perimeter_mean', 'area_mean', 
                       'smoothness_mean', 'compactness_mean', 
                       'concavity_mean','concave_points_mean', 
                       'symmetry_mean', 'fractal_dimension_mean',
                       'radius_se', 'texture_se', 'perimeter_se', 
                       'area_se', 'smoothness_se', 'compactness_se', 
                       'concavity_se', 'concave_points_se', 
                       'symmetry_se', 'fractal_dimension_se', 
                       'radius_worst', 'texture_worst', 
                       'perimeter_worst', 'area_worst', 
                       'smoothness_worst', 'compactness_worst', 
                       'concavity_worst', 'concave_points_worst', 
                       'symmetry_worst', 'fractal_dimension_worst')
Dat.cancer$id_number <- NULL
table(Dat.cancer$diagnosis)

DF <- Dat.cancer %>%
  mutate(Y = ifelse(diagnosis == "B", 0, ifelse(diagnosis == "M", 1, NA)),
         diagnosis = as.factor(Y)) %>%
  dplyr::select(-diagnosis) %>% 
  dplyr::select(Y, everything())


####################################################################
############### Interaction Selection Algorithms ###################
####################################################################

LASSO.func <- function(train.DF) {
  models = list()
  Runtime = list()
  LASSO.time <- system.time(LASSO.Mod <- cv.glmnet(x = as.matrix(train.DF[,-1]), y = factor(train.DF$Y),
                                                   family = "binomial", type.measure = "class", alpha = 1))
  
  models[["lasso"]] = LASSO.Mod
  Runtime = LASSO.time
  return(list(models, Runtime))
}

SCAD.func <- function(train.DF) {
  models = list()
  Runtime = list()
  SCAD.time <- system.time(SCAD.Mod <- cv.ncvreg(X = as.matrix(train.DF[,-1]), y = factor(train.DF$Y),
                                                 family = "binomial", penalty = "SCAD", nfolds = 10))
  
  models[["scad"]] = SCAD.Mod
  Runtime = SCAD.time
  return(list(models, Runtime))
}

MCP.func <- function(train.DF) {
  models = list()
  Runtime = list()
  MCP.time <- system.time(MCP.Mod <- cv.ncvreg(X = as.matrix(train.DF[,-1]), y = factor(train.DF$Y),
                                               family = "binomial", penalty = "MCP", nfolds = 10))
  
  models[["mcp"]] = MCP.Mod
  Runtime = MCP.time
  return(list(models, Runtime))
}

RLW.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Weak.time <- system.time(RAMP.LASSO_w <- RAMP(as.matrix(train.DF[,-1]), train.DF$Y, family = "binomial",
                                                     penalty = "LASSO", gamma = NULL, inter = TRUE, hier = "Weak",
                                                     eps = 1e-15, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                     inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                     ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RLW"]] = RAMP.LASSO_w
  Runtime = RAMP_Weak.time
  return(list(models, Runtime))
}

RLS.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Strong.time <- system.time(RAMP.LASSO_s <- RAMP(as.matrix(train.DF[,-1]), train.DF$Y, family = "binomial",
                                                       penalty = "LASSO", gamma = NULL, inter = TRUE, hier = "Strong",
                                                       eps = 1e-15, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                       inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                       ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RLS"]] = RAMP.LASSO_s
  Runtime = RAMP_Strong.time
  return(list(models, Runtime))
}

RSW.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Weak.time <- system.time(RAMP.SCAD_w <- RAMP(as.matrix(train.DF[,-1]), train.DF$Y, family = "binomial",
                                                    penalty = "SCAD", gamma = NULL, inter = TRUE, hier = "Weak",
                                                    eps = 1e-15, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                    inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                    ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RSW"]] = RAMP.SCAD_w
  Runtime = RAMP_Weak.time
  return(list(models, Runtime))
}

RSS.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Strong.time <- system.time(RAMP.SCAD_s <- RAMP(as.matrix(train.DF[,-1]), train.DF$Y, family = "binomial",
                                                      penalty = "SCAD", gamma = NULL, inter = TRUE, hier = "Strong",
                                                      eps = 1e-15, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                      inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                      ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RSS"]] = RAMP.SCAD_s
  Runtime = RAMP_Strong.time
  return(list(models, Runtime))
}

RMCPW.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Weak.time <- system.time(RAMP.MCP_w <- RAMP(as.matrix(train.DF[,-1]), train.DF$Y, family = "binomial",
                                                   penalty = "MCP", gamma = NULL, inter = TRUE, hier = "Weak",
                                                   eps = 1e-15, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                   inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                   ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RMCPW"]] = RAMP.MCP_w
  Runtime = RAMP_Weak.time
  return(list(models, Runtime))
}

RMCPS.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Strong.time <- system.time(RAMP.MCP_s <- RAMP(as.matrix(train.DF[,-1]), train.DF$Y, family = "binomial",
                                                     penalty = "MCP", gamma = NULL, inter = TRUE, hier = "Strong",
                                                     eps = 1e-15, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                     inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                     ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RMCPS"]] = RAMP.MCP_s
  Runtime = RAMP_Strong.time
  return(list(models, Runtime))
}

iRF.func <- function(train.DF) {
  models.iRF = list()
  Runtime = list()
  
  iRF.Time <- system.time({
    Mod.iRF <- iRF(as.matrix(train.DF[,-1]), factor(train.DF$Y), xtest = NULL, ytest = NULL,
                   n.iter = 5,
                   ntree = 500,
                   n.core = 5,
                   mtry.select.prob = rep(1/ncol(train.DF[,-1]), ncol(train.DF[,-1])),
                   keep.impvar.quantile = NULL,
                   interactions.return = c(5),
                   wt.pred.accuracy = FALSE,
                   cutoff.unimp.feature = 0,
                   rit.param = list(depth = 20, ntree = 100, nchild = 2, class.id = 1, class.cut = NULL),
                   varnames.grp = NULL,
                   n.bootstrap = 30,
                   bootstrap.forest = TRUE,
                   verbose = TRUE
    )
  })
  
  models.iRF[["iRF"]] = Mod.iRF
  Runtime = iRF.Time
  return(list(models.iRF, Runtime))
}

RealBal.func <- function(n, nsim){
  #To save the simulation results
  CM.LASSO <- list(NULL)
  CM.SCAD <- list(NULL)
  CM.MCP <- list(NULL)
  CM.LW <- list(NULL)
  CM.LS <- list(NULL)
  CM.SW <- list(NULL)
  CM.SS <- list(NULL)
  CM.MCPW <- list(NULL)
  CM.MCPS <- list(NULL)
  MCC.ALL <- list(NULL)
  CM.iRF <- list(NULL)
  MI.ALL <- list(NULL)
  IInd.ALL <- list(NULL)
  MC.ALL <- list(NULL)
  IntCoef.ALL <- list(NULL)
  
  iter <- seq(1,nsim)
  simClass.f <- function(iter){
    # generating data and splitting to train/test
    train.id = sample(n, floor(0.60*n), replace = FALSE, prob = NULL)
    train.DF = DF[train.id, ]
    test.DF = DF[-train.id, ]
    
    LASSO.res.Bin = LASSO.func(train.DF)
    SCAD.res.Bin = SCAD.func(train.DF)
    MCP.res.Bin = MCP.func(train.DF)
    RAMP.LW.Bin = RLW.func(train.DF)
    RAMP.LS.Bin = RLS.func(train.DF)
    RAMP.SW.Bin = RSW.func(train.DF)
    RAMP.SS.Bin = RSS.func(train.DF)
    RAMP.MW.Bin = RMCPW.func(train.DF)
    RAMP.MS.Bin = RMCPS.func(train.DF)
    iRF.Bin = iRF.func(train.DF)
    
    ## Evaluating Interaction Selection Performance by Computing Beta Sensitivity and Specificity ##
    
    MEI.RLW = RAMP.LW.Bin[[1]]$RLW$mainInd # Index for the selected Main Effects.
    IInd.RLW = RAMP.LW.Bin[[1]]$RLW$interInd # Index for the selected Interaction Effects
    MECoef.RLW = RAMP.LW.Bin[[1]]$RLW$beta.m # Coefficients for the selected Main Effects
    IntCoef.RLW = RAMP.LW.Bin[[1]]$RLW$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RLW) = IInd.RLW
    
    MEI.RLS = RAMP.LS.Bin[[1]]$RLS$mainInd # Index for the selected Main Effects.
    IInd.RLS = RAMP.LS.Bin[[1]]$RLS$interInd # Index for the selected Interaction Effects
    MECoef.RLS = RAMP.LS.Bin[[1]]$RLS$beta.m # Coefficients for the selected Main Effects
    IntCoef.RLS = RAMP.LS.Bin[[1]]$RLS$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RLS) = IInd.RLS
    
    MEI.RSW = RAMP.SW.Bin[[1]]$RSW$mainInd # Index for the selected Main Effects.
    IInd.RSW = RAMP.SW.Bin[[1]]$RSW$interInd # Index for the selected Interaction Effects
    MECoef.RSW = RAMP.SW.Bin[[1]]$RSW$beta.m # Coefficients for the selected Main Effects
    IntCoef.RSW = RAMP.SW.Bin[[1]]$RSW$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RSW) = IInd.RSW
    
    MEI.RSS = RAMP.SS.Bin[[1]]$RSS$mainInd # Index for the selected Main Effects.
    IInd.RSS = RAMP.SS.Bin[[1]]$RSS$interInd # Index for the selected Interaction Effects
    MECoef.RSS = RAMP.SS.Bin[[1]]$RSS$beta.m # Coefficients for the selected Main Effects
    IntCoef.RSS = RAMP.SS.Bin[[1]]$RSS$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RSS) = IInd.RSS
    
    MEI.RMCPW = RAMP.MW.Bin[[1]]$RMCPW$mainInd # Index for the selected Main Effects.
    IInd.RMCPW = RAMP.MW.Bin[[1]]$RMCPW$interInd # Index for the selected Interaction Effects
    MECoef.RMCPW = RAMP.MW.Bin[[1]]$RMCPW$beta.m # Coefficients for the selected Main Effects
    IntCoef.RMCPW = RAMP.MW.Bin[[1]]$RMCPW$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RMCPW) = IInd.RMCPW
    
    MEI.RMCPS = RAMP.MS.Bin[[1]]$RMCPS$mainInd # Index for the selected Main Effects.
    IInd.RMCPS = RAMP.MS.Bin[[1]]$RMCPS$interInd # Index for the selected Interaction Effects
    MECoef.RMCPS = RAMP.MS.Bin[[1]]$RMCPS$beta.m # Coefficients for the selected Main Effects
    IntCoef.RMCPS = RAMP.MS.Bin[[1]]$RMCPS$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RMCPS) = IInd.RMCPS
    
    IntCoef.iRF = iRF.Bin[[1]][['iRF']]$interaction[[5]]
    IntCoef.iRF.Sig = IntCoef.iRF[IntCoef.iRF>=0.7]
    
    # Evaluating Interaction Selection Performance by Computing Confusion Matrix to report Accuracy, F1-score, Sensitivity and Specificity
    pred.LASSO = predict(LASSO.res.Bin[[1]]$lasso, as.matrix(test.DF[,-1]), type = "class", s = "lambda.min")
    pred.SCAD = predict(SCAD.res.Bin[[1]]$scad, as.matrix(test.DF[,-1]), type = "class")
    pred.MCP = predict(MCP.res.Bin[[1]]$mcp, as.matrix(test.DF[,-1]), type = "class")
    pred.RLW = predict(RAMP.LW.Bin[[1]][["RLW"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RLS <- predict(RAMP.LS.Bin[[1]][["RLS"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RSW <- predict(RAMP.SW.Bin[[1]][["RSW"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RSS <- predict(RAMP.SS.Bin[[1]][["RSS"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RMW <- predict(RAMP.MW.Bin[[1]][["RMCPW"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RMS <- predict(RAMP.MS.Bin[[1]][["RMCPS"]], as.matrix(test.DF[,-1]), type = "class")
    pred.iRF <- predict(iRF.Bin[[1]][["iRF"]]$rf.list, as.matrix(test.DF[,-1]), type = "class")
    
    yhat.RLW <- as.integer(pred.RLW)
    yhat.RLS <- as.integer(pred.RLS)
    yhat.RSW <- as.integer(pred.RSW)
    yhat.RSS <- as.integer(pred.RSS)
    yhat.RMW <- as.integer(pred.RMW)
    yhat.RMS <- as.integer(pred.RMS)
    
    CM.BinLASSO <- confusionMatrix(factor(pred.LASSO), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.LASSO = round(mccr(test.DF$Y, factor(pred.LASSO)), 4)
    CM.BinSCAD <- confusionMatrix(factor(pred.SCAD), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.SCAD = round(mccr(test.DF$Y, factor(pred.SCAD)), 4)
    CM.BinMCP <- confusionMatrix(factor(pred.MCP), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.MCP = round(mccr(test.DF$Y, factor(pred.MCP)), 4)
    CM.BinRLW <- confusionMatrix(factor(yhat.RLW), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RLW = round(mccr(test.DF$Y, factor(yhat.RLW)), 4)
    CM.BinRLS <- confusionMatrix(factor(yhat.RLS), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RLS = round(mccr(test.DF$Y, factor(yhat.RLS)), 4)
    CM.BinRSW <- confusionMatrix(factor(yhat.RSW), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RSW = round(mccr(test.DF$Y, factor(yhat.RSW)), 4)
    CM.BinRSS <- confusionMatrix(factor(yhat.RSS), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RSS = round(mccr(test.DF$Y, factor(yhat.RSS)), 4)
    CM.BinRMW <- confusionMatrix(factor(yhat.RMW), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RMW = round(mccr(test.DF$Y, factor(yhat.RMW)), 4)
    CM.BinRMS <- confusionMatrix(factor(yhat.RMS), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RMS = round(mccr(test.DF$Y, factor(yhat.RMS)), 4)
    
    CM.BiniRF1 <- confusionMatrix(factor(pred.iRF[[1]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF1 = round(mccr(test.DF$Y, pred.iRF[[1]]), 4)
    CM.BiniRF2 <- confusionMatrix(factor(pred.iRF[[2]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF2 = round(mccr(test.DF$Y, pred.iRF[[2]]), 4)
    CM.BiniRF3 <- confusionMatrix(factor(pred.iRF[[3]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF3 = round(mccr(test.DF$Y, pred.iRF[[3]]), 4)
    CM.BiniRF4 <- confusionMatrix(factor(pred.iRF[[4]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF4 = round(mccr(test.DF$Y, pred.iRF[[4]]), 4)
    CM.BiniRF5 <- confusionMatrix(factor(pred.iRF[[5]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF5 = round(mccr(test.DF$Y, pred.iRF[[5]]), 4)
    
    
    Metrics <- list(MEI.RLW, MEI.RLS, MEI.RSW, MEI.RSS, MEI.RMCPW, MEI.RMCPS, IInd.RLW, IInd.RLS, IInd.RSW, IInd.RSS, IInd.RMCPW, IInd.RMCPS, 
                    MECoef.RLW, MECoef.RLS, MECoef.RSW, MECoef.RSS, MECoef.RMCPW, MECoef.RMCPS, IntCoef.RLW, IntCoef.RLS, IntCoef.RSW, IntCoef.RSS,
                    IntCoef.RMCPW, IntCoef.RMCPS, IntCoef.iRF.Sig, CM.BinLASSO, MCC.LASSO, CM.BinSCAD, MCC.SCAD, CM.BinMCP, MCC.MCP, CM.BinRLW, MCC.RLW,
                    CM.BinRLS, MCC.RLS, CM.BinRSW, MCC.RSW, CM.BinRSS, MCC.RSS, CM.BinRMW, MCC.RMW, CM.BinRMS, MCC.RMS, CM.BiniRF1, MCC.iRF1, CM.BiniRF2,
                    MCC.iRF2, CM.BiniRF3, MCC.iRF3, CM.BiniRF4, MCC.iRF4, CM.BiniRF5, MCC.iRF5)
    
    return(Metrics)
    
  }
  
  numCores <- detectCores()
  results <- mclapply(iter, simClass.f, mc.cores = 6)
  
  #Collect values of estimates
  MEI_RLW = (do.call(cbind, results)[1,])
  MEI_RLS = (do.call(cbind, results)[2,])
  MEI_RSW = (do.call(cbind, results)[3,])
  MEI_RSS = (do.call(cbind, results)[4,])
  MEI_RMCPW = (do.call(cbind, results)[5,])
  MEI_RMCPS = (do.call(cbind, results)[6,])
  
  IInd_RLW = (do.call(cbind, results)[7,])
  IInd_RLS = (do.call(cbind, results)[8,])
  IInd_RSW = (do.call(cbind, results)[9,])
  IInd_RSS = (do.call(cbind, results)[10,])
  IInd_RMCPW = (do.call(cbind, results)[11,])
  IInd_RMCPS = (do.call(cbind, results)[12,])
  
  MECoef_RLW = (do.call(cbind, results)[13,])
  MECoef_RLS = (do.call(cbind, results)[14,])
  MECoef_RSW = (do.call(cbind, results)[15,])
  MECoef_RSS = (do.call(cbind, results)[16,])
  MECoef_RMCPW = (do.call(cbind, results)[17,])
  MECoef_RMCPS = (do.call(cbind, results)[18,])
  
  IntCoef_RLW = (do.call(cbind, results)[19,])
  IntCoef_RLS = (do.call(cbind, results)[20,])
  IntCoef_RSW = (do.call(cbind, results)[21,])
  IntCoef_RSS = (do.call(cbind, results)[22,])
  IntCoef_RMCPW = (do.call(cbind, results)[23,])
  IntCoef_RMCPS = (do.call(cbind, results)[24,])
  IntCoef_iRF.Sig = (do.call(cbind, results)[25,])
  
  CM_LASSO = (do.call(cbind, results)[26,])
  MCC_LASSO = (do.call(cbind, results)[27,])
  CM_SCAD = (do.call(cbind, results)[28,])
  MCC_SCAD = (do.call(cbind, results)[29,])
  CM_MCP = (do.call(cbind, results)[30,])
  MCC_MCP = (do.call(cbind, results)[31,])
  CM_RLW = (do.call(cbind, results)[32,])
  MCC_RLW = (do.call(cbind, results)[33,])
  CM_RLS = (do.call(cbind, results)[34,])
  MCC_RLS = (do.call(cbind, results)[35,])
  CM_RSW = (do.call(cbind, results)[36,])
  MCC_RSW = (do.call(cbind, results)[37,])
  CM_RSS = (do.call(cbind, results)[38,])
  MCC_RSS = (do.call(cbind, results)[39,])
  CM_RMCPW = (do.call(cbind, results)[40,])
  MCC_RMW = (do.call(cbind, results)[41,])
  CM_RMCPS = (do.call(cbind, results)[42,])
  MCC_RMS = (do.call(cbind, results)[43,])
  
  CM_iRF1 = (do.call(cbind, results)[44,])
  MCC_iRF1 = (do.call(cbind, results)[45,])
  CM_iRF2 = (do.call(cbind, results)[46,])
  MCC_iRF2 = (do.call(cbind, results)[47,])
  CM_iRF3 = (do.call(cbind, results)[48,])
  MCC_iRF3 = (do.call(cbind, results)[49,])
  CM_iRF4 = (do.call(cbind, results)[50,])
  MCC_iRF4 = (do.call(cbind, results)[51,])
  CM_iRF5 = (do.call(cbind, results)[52,])
  MCC_iRF5 = (do.call(cbind, results)[53,])
  
  CM.LASSO <- list(CM_LASSO)
  CM.SCAD <- list(CM_SCAD)
  CM.MCP <- list(CM_MCP)
  CM.LW = list(CM_RLW)
  CM.LS = list(CM_RLS)
  CM.SW = list(CM_RSW)
  CM.SS = list(CM_RSS)
  CM.MCPW = list(CM_RMCPW)
  CM.MCPS = list(CM_RMCPS)
  CM.iRF = list(CM_iRF1, CM_iRF2, CM_iRF3, CM_iRF4, CM_iRF5)
  
  MCC.ALL = list(MCC_LASSO, MCC_SCAD, MCC_MCP, MCC_RLW, MCC_RLS, MCC_RSW, MCC_RSS, MCC_RMW, MCC_RMS, MCC_iRF1, MCC_iRF2, MCC_iRF3, MCC_iRF4, MCC_iRF5)
  MI.ALL = list(MEI_RLW, MEI_RLS, MEI_RSW, MEI_RSS, MEI_RMCPW, MEI_RMCPS)
  IInd.ALL = list(IInd_RLW, IInd_RLS, IInd_RSW, IInd_RSS, IInd_RMCPW, IInd_RMCPS) 
  MC.ALL = list(MECoef_RLW, MECoef_RLS, MECoef_RSW, MECoef_RSS, MECoef_RMCPW, MECoef_RMCPS)
  IntCoef.ALL = list(IntCoef_RLW, IntCoef_RLS, IntCoef_RSW, IntCoef_RSS, IntCoef_RMCPW, IntCoef_RMCPS, IntCoef_iRF.Sig)
  
  result.final1 <- list(CM.LASSO, CM.SCAD, CM.MCP, CM.LW, CM.LS, CM.SW, CM.SS, CM.MCPW, CM.MCPS, CM.iRF, MCC.ALL)
  
  
  result.final2 <- list(MI.ALL, IInd.ALL, MC.ALL, IntCoef.ALL)
  
  return(list(result.final1, result.final2))
}

set.seed(1788)

n <- nrow(DF)
nsim <- 100
system.time(RealBal <- RealBal.func(n,nsim))


###########################################################################################
######## Evaluating Performance by Computing Accuracy, Sensitivity, Specificity ###########
######## F1-Score, Balanced Accuracy and Matthew's Correlation Coefficients ###############
###########################################################################################
LAS.Acc <- matrix(NA, nrow = nsim, ncol = 1)
LAS.Sens <- matrix(NA, nrow = nsim, ncol = 1)
LAS.Spec <- matrix(NA, nrow = nsim, ncol = 1)
LAS.F1 <- matrix(NA, nrow = nsim, ncol = 1)
LAS.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

SCAD.Acc <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.Sens <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.Spec <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.F1 <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

MCP.Acc <- matrix(NA, nrow = nsim, ncol = 1)
MCP.Sens <- matrix(NA, nrow = nsim, ncol = 1)
MCP.Spec <- matrix(NA, nrow = nsim, ncol = 1)
MCP.F1 <- matrix(NA, nrow = nsim, ncol = 1)
MCP.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RLW.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RLW.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RLW.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RLW.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RLW.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RLS.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RLS.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RLS.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RLS.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RLS.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RSW.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RSW.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RSW.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RSW.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RSW.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RSS.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RSS.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RSS.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RSS.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RSS.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RMCPW.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RMCPS.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF1.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF2.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF3.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF4.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF5.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

LAS.MCC <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.MCC <- matrix(NA, nrow = nsim, ncol = 1)
MCP.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RLW.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RLS.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RSW.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RSS.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.MCC <- matrix(NA, nrow = nsim, ncol = 1)

for (i in 1:nsim) {
  LAS.Acc[i,] = RealBal[[1]][[1]][[1]][[i]]$overall[1]
  LAS.Sens[i,] = RealBal[[1]][[1]][[1]][[i]]$byClass[1]
  LAS.Spec[i,] = RealBal[[1]][[1]][[1]][[i]]$byClass[2]
  LAS.F1[i,] = RealBal[[1]][[1]][[1]][[i]]$byClass[7]
  LAS.BalAcc[i,] = RealBal[[1]][[1]][[1]][[i]]$byClass[11]
  
  SCAD.Acc[i,] = RealBal[[1]][[2]][[1]][[i]]$overall[1]
  SCAD.Sens[i,] = RealBal[[1]][[2]][[1]][[i]]$byClass[1]
  SCAD.Spec[i,] = RealBal[[1]][[2]][[1]][[i]]$byClass[2]
  SCAD.F1[i,] = RealBal[[1]][[2]][[1]][[i]]$byClass[7]
  SCAD.BalAcc[i,] = RealBal[[1]][[2]][[1]][[i]]$byClass[11]
  
  MCP.Acc[i,] = RealBal[[1]][[3]][[1]][[i]]$overall[1]
  MCP.Sens[i,] = RealBal[[1]][[3]][[1]][[i]]$byClass[1]
  MCP.Spec[i,] = RealBal[[1]][[3]][[1]][[i]]$byClass[2]
  MCP.F1[i,] = RealBal[[1]][[3]][[1]][[i]]$byClass[7]
  MCP.BalAcc[i,] = RealBal[[1]][[3]][[1]][[i]]$byClass[11]
  
  RLW.Acc[i,] = RealBal[[1]][[4]][[1]][[i]]$overall[1]
  RLW.Sens[i,] = RealBal[[1]][[4]][[1]][[i]]$byClass[1]
  RLW.Spec[i,] = RealBal[[1]][[4]][[1]][[i]]$byClass[2]
  RLW.F1[i,] = RealBal[[1]][[4]][[1]][[i]]$byClass[7]
  RLW.BalAcc[i,] = RealBal[[1]][[4]][[1]][[i]]$byClass[11]
  
  RLS.Acc[i,] = RealBal[[1]][[5]][[1]][[i]]$overall[1]
  RLS.Sens[i,] = RealBal[[1]][[5]][[1]][[i]]$byClass[1]
  RLS.Spec[i,] = RealBal[[1]][[5]][[1]][[i]]$byClass[2]
  RLS.F1[i,] = RealBal[[1]][[5]][[1]][[i]]$byClass[7]
  RLS.BalAcc[i,] = RealBal[[1]][[5]][[1]][[i]]$byClass[11]
  
  RSW.Acc[i,] = RealBal[[1]][[6]][[1]][[i]]$overall[1]
  RSW.Sens[i,] = RealBal[[1]][[6]][[1]][[i]]$byClass[1]
  RSW.Spec[i,] = RealBal[[1]][[6]][[1]][[i]]$byClass[2]
  RSW.F1[i,] = RealBal[[1]][[6]][[1]][[i]]$byClass[7]
  RSW.BalAcc[i,] = RealBal[[1]][[6]][[1]][[i]]$byClass[11]
  
  RSS.Acc[i,] = RealBal[[1]][[7]][[1]][[i]]$overall[1]
  RSS.Sens[i,] = RealBal[[1]][[7]][[1]][[i]]$byClass[1]
  RSS.Spec[i,] = RealBal[[1]][[7]][[1]][[i]]$byClass[2]
  RSS.F1[i,] = RealBal[[1]][[7]][[1]][[i]]$byClass[7]
  RSS.BalAcc[i,] = RealBal[[1]][[7]][[1]][[i]]$byClass[11]
  
  RMCPW.Acc[i,] = RealBal[[1]][[8]][[1]][[i]]$overall[1]
  RMCPW.Sens[i,] = RealBal[[1]][[8]][[1]][[i]]$byClass[1]
  RMCPW.Spec[i,] = RealBal[[1]][[8]][[1]][[i]]$byClass[2]
  RMCPW.F1[i,] = RealBal[[1]][[8]][[1]][[i]]$byClass[7]
  RMCPW.BalAcc[i,] = RealBal[[1]][[8]][[1]][[i]]$byClass[11]
  
  RMCPS.Acc[i,] = RealBal[[1]][[9]][[1]][[i]]$overall[1]
  RMCPS.Sens[i,] = RealBal[[1]][[9]][[1]][[i]]$byClass[1]
  RMCPS.Spec[i,] = RealBal[[1]][[9]][[1]][[i]]$byClass[2]
  RMCPS.F1[i,] = RealBal[[1]][[9]][[1]][[i]]$byClass[7]
  RMCPS.BalAcc[i,] = RealBal[[1]][[9]][[1]][[i]]$byClass[11]
  
  iRF1.Acc[i,] = RealBal[[1]][[10]][[1]][[i]]$overall[1]
  iRF1.Sens[i,] = RealBal[[1]][[10]][[1]][[i]]$byClass[1]
  iRF1.Spec[i,] = RealBal[[1]][[10]][[1]][[i]]$byClass[2]
  iRF1.F1[i,] = RealBal[[1]][[10]][[1]][[i]]$byClass[7]
  iRF1.BalAcc[i,] = RealBal[[1]][[10]][[1]][[i]]$byClass[11]
  
  iRF2.Acc[i,] = RealBal[[1]][[10]][[2]][[i]]$overall[1]
  iRF2.Sens[i,] = RealBal[[1]][[10]][[2]][[i]]$byClass[1]
  iRF2.Spec[i,] = RealBal[[1]][[10]][[2]][[i]]$byClass[2]
  iRF2.F1[i,] = RealBal[[1]][[10]][[2]][[i]]$byClass[7]
  iRF2.BalAcc[i,] = RealBal[[1]][[10]][[2]][[i]]$byClass[11]
  
  iRF3.Acc[i,] = RealBal[[1]][[10]][[3]][[i]]$overall[1]
  iRF3.Sens[i,] = RealBal[[1]][[10]][[3]][[i]]$byClass[1]
  iRF3.Spec[i,] = RealBal[[1]][[10]][[3]][[i]]$byClass[2]
  iRF3.F1[i,] = RealBal[[1]][[10]][[3]][[i]]$byClass[7]
  iRF3.BalAcc[i,] = RealBal[[1]][[10]][[3]][[i]]$byClass[11]
  
  iRF4.Acc[i,] = RealBal[[1]][[10]][[4]][[i]]$overall[1]
  iRF4.Sens[i,] = RealBal[[1]][[10]][[4]][[i]]$byClass[1]
  iRF4.Spec[i,] = RealBal[[1]][[10]][[4]][[i]]$byClass[2]
  iRF4.F1[i,] = RealBal[[1]][[10]][[4]][[i]]$byClass[7]
  iRF4.BalAcc[i,] = RealBal[[1]][[10]][[4]][[i]]$byClass[11]
  
  iRF5.Acc[i,] = RealBal[[1]][[10]][[5]][[i]]$overall[1]
  iRF5.Sens[i,] = RealBal[[1]][[10]][[5]][[i]]$byClass[1]
  iRF5.Spec[i,] = RealBal[[1]][[10]][[5]][[i]]$byClass[2]
  iRF5.F1[i,] = RealBal[[1]][[10]][[5]][[i]]$byClass[7]
  iRF5.BalAcc[i,] = RealBal[[1]][[10]][[5]][[i]]$byClass[11]
  
  LAS.MCC[i,] = RealBal[[1]][[11]][[1]][[i]]
  SCAD.MCC[i,] = RealBal[[1]][[11]][[2]][[i]]
  MCP.MCC[i,] = RealBal[[1]][[11]][[3]][[i]]
  RLW.MCC[i,] = RealBal[[1]][[11]][[4]][[i]]
  RLS.MCC[i,] = RealBal[[1]][[11]][[5]][[i]]
  RSW.MCC[i,] = RealBal[[1]][[11]][[6]][[i]]
  RSS.MCC[i,] = RealBal[[1]][[11]][[7]][[i]]
  RMCPW.MCC[i,] = RealBal[[1]][[11]][[8]][[i]]
  RMCPS.MCC[i,] = RealBal[[1]][[11]][[9]][[i]]
  iRF1.MCC[i,] = RealBal[[1]][[11]][[10]][[i]]
  iRF2.MCC[i,] = RealBal[[1]][[11]][[11]][[i]]
  iRF3.MCC[i,] = RealBal[[1]][[11]][[12]][[i]]
  iRF4.MCC[i,] = RealBal[[1]][[11]][[13]][[i]]
  iRF5.MCC[i,] = RealBal[[1]][[11]][[14]][[i]]
  
  Pred.Result = cbind(LAS.Acc, LAS.Sens, LAS.Spec, LAS.F1, LAS.BalAcc, LAS.MCC,
                      SCAD.Acc, SCAD.Sens, SCAD.Spec, SCAD.F1, SCAD.BalAcc, SCAD.MCC,
                      MCP.Acc, MCP.Sens, MCP.Spec, MCP.F1, MCP.BalAcc, MCP.MCC,
                      RLW.Acc, RLW.Sens, RLW.Spec, RLW.F1, RLW.BalAcc, RLW.MCC,
                      RLS.Acc, RLS.Sens, RLS.Spec, RLS.F1, RLS.BalAcc, RLS.MCC,
                      RSW.Acc, RSW.Sens, RSW.Spec, RSW.F1, RSW.BalAcc, RSW.MCC,
                      RSS.Acc, RSS.Sens, RSS.Spec, RSS.F1, RSS.BalAcc, RSS.MCC,
                      RMCPW.Acc, RMCPW.Sens, RMCPW.Spec, RMCPW.F1, RMCPW.BalAcc, RMCPW.MCC,
                      RMCPS.Acc, RMCPS.Sens, RMCPS.Spec, RMCPS.F1, RMCPS.BalAcc, RMCPS.MCC,
                      iRF1.Acc, iRF1.Sens, iRF1.Spec, iRF1.F1, iRF1.BalAcc, iRF1.MCC,
                      iRF2.Acc, iRF2.Sens, iRF2.Spec, iRF2.F1, iRF2.BalAcc, iRF2.MCC,
                      iRF3.Acc, iRF3.Sens, iRF3.Spec, iRF3.F1, iRF3.BalAcc, iRF3.MCC,
                      iRF4.Acc, iRF4.Sens, iRF4.Spec, iRF4.F1, iRF4.BalAcc, iRF4.MCC,
                      iRF5.Acc, iRF5.Sens, iRF5.Spec, iRF5.F1, iRF5.BalAcc, iRF5.MCC)
}

colnames(Pred.Result) = c("LAS.Acc", "LAS.Sens", "LAS.Spec", "LAS.F1", "LAS.BalAcc", "LAS.MCC",
                          "SCAD.Acc", "SCAD.Sens", "SCAD.Spec", "SCAD.F1", "SCAD.BalAcc", "SCAD.MCC",
                          "MCP.Acc", "MCP.Sens", "MCP.Spec", "MCP.F1", "MCP.BalAcc", "MCP.MCC",
                          "RLW.Acc", "RLW.Sens", "RLW.Spec", "RLW.F1", "RLW.BalAcc", "RLW.MCC",
                          "RLS.Acc", "RLS.Sens", "RLS.Spec", "RLS.F1", "RLS.BalAcc", "RLS.MCC",
                          "RSW.Acc", "RSW.Sens", "RSW.Spec", "RSW.F1", "RSW.BalAcc", "RSW.MCC",
                          "RSS.Acc", "RSS.Sens", "RSS.Spec", "RSS.F1", "RSS.BalAcc", "RSS.MCC",
                          "RMCPW.Acc", "RMCPW.Sens", "RMCPW.Spec", "RMCPW.F1", "RMCPW.BalAcc", "RMCPW.MCC",
                          "RMCPS.Acc", "RMCPS.Sens", "RMCPS.Spec", "RMCPS.F1", "RMCPS.BalAcc", "RMCPS.MCC",
                          "iRF1.Acc", "iRF1.Sens", "iRF1.Spec", "iRF1.F1", "iRF1.BalAcc", "iRF1.MCC",
                          "iRF2.Acc", "iRF2.Sens", "iRF2.Spec", "iRF2.F1", "iRF2.BalAcc", "iRF2.MCC",
                          "iRF3.Acc", "iRF3.Sens", "iRF3.Spec", "iRF3.F1", "iRF3.BalAcc", "iRF3.MCC",
                          "iRF4.Acc", "iRF4.Sens", "iRF4.Spec", "iRF4.F1", "iRF4.BalAcc", "iRF4.MCC",
                          "iRF5.Acc", "iRF5.Sens", "iRF5.Spec", "iRF5.F1", "iRF5.BalAcc", "iRF5.MCC")
Pred.Real = Pred.Result

PredResult <- as.data.frame(Pred.Real)

PredMetrics <- PredResult %>% 
  dplyr::summarise_each(funs(mean, sd, SE = sd(.)/sqrt(n())))

ClassReal <- PredMetrics %>% 
  tidyr::gather(key = Pred, value = value) 
ClassReal.sep <- separate(ClassReal,
                          col = Pred,
                          sep = "_",
                          into = c("Model", "Metrics"))

Metrics.ClassReal <- ClassReal.sep %>% 
  spread(key = Metrics, value = value)

####################################################################
########## Imbalanced Wisconsin Breast Cancer Data #################
####################################################################

DF.1 <- read_table("Features.txt", col_names = FALSE)
DF.2 <- read_table("Info.txt", col_names = FALSE)

DF.3 <- DF.2 %>% 
  dplyr::rename(BC = X1) %>% 
  dplyr::select(BC)

DF.Full <- bind_cols(DF.3, DF.1) %>% 
  dplyr::select(-X118)

DF.Imb <- DF.Full %>%
  dplyr::select(BC, contains("X"))%>%
  mutate(Y = ifelse(BC == "-1", 0, ifelse(BC == "1", 1, NA))) %>%
  dplyr::select(-BC)%>% 
  dplyr::select(Y, everything())

####################################################################
############### Interaction Selection Algorithms ###################
####################################################################

LASSO.func <- function(train.DF) {
  models = list()
  Runtime = list()
  LASSO.time <- system.time(LASSO.Mod <- cv.glmnet(x = as.matrix(train.DF[,-1]), y = as.factor(train.DF$Y),
                                                   family = "binomial", type.measure = "class", alpha = 1))
  
  models[["lasso"]] = LASSO.Mod
  Runtime = LASSO.time
  return(list(models, Runtime))
}

SCAD.func <- function(train.DF) {
  models = list()
  Runtime = list()
  SCAD.time <- system.time(SCAD.Mod <- cv.ncvreg(X = as.matrix(train.DF[,-1]), y = factor(train.DF$Y),
                                                 family = "binomial", penalty = "SCAD", nfolds = 10))
  
  models[["scad"]] = SCAD.Mod
  Runtime = SCAD.time
  return(list(models, Runtime))
}

MCP.func <- function(train.DF) {
  models = list()
  Runtime = list()
  MCP.time <- system.time(MCP.Mod <- cv.ncvreg(X = as.matrix(train.DF[,-1]), y = factor(train.DF$Y),
                                               family = "binomial", penalty = "MCP", nfolds = 10))
  
  models[["mcp"]] = MCP.Mod
  Runtime = MCP.time
  return(list(models, Runtime))
}

RLW.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Weak.time <- system.time(RAMP.LASSO_w <- RAMP(as.matrix(train.DF[,-1]), as.numeric(train.DF$Y), family = "binomial",
                                                     penalty = "LASSO", gamma = NULL, inter = TRUE, hier = "Weak",
                                                     eps = 1e-10, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                     inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                     ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RLW"]] = RAMP.LASSO_w
  Runtime = RAMP_Weak.time
  return(list(models, Runtime))
}

RLS.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Strong.time <- system.time(RAMP.LASSO_s <- RAMP(as.matrix(train.DF[,-1]), as.numeric(train.DF$Y), family = "binomial",
                                                       penalty = "LASSO", gamma = NULL, inter = TRUE, hier = "Strong",
                                                       eps = 1e-10, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                       inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                       ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RLS"]] = RAMP.LASSO_s
  Runtime = RAMP_Strong.time
  return(list(models, Runtime))
}

RSW.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Weak.time <- system.time(RAMP.SCAD_w <- RAMP(as.matrix(train.DF[,-1]), as.numeric(train.DF$Y), family = "binomial",
                                                    penalty = "SCAD", gamma = NULL, inter = TRUE, hier = "Weak",
                                                    eps = 1e-10, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                    inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                    ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RSW"]] = RAMP.SCAD_w
  Runtime = RAMP_Weak.time
  return(list(models, Runtime))
}

RSS.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Strong.time <- system.time(RAMP.SCAD_s <- RAMP(as.matrix(train.DF[,-1]), as.numeric(train.DF$Y), family = "binomial",
                                                      penalty = "SCAD", gamma = NULL, inter = TRUE, hier = "Strong",
                                                      eps = 1e-10, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                      inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                      ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RSS"]] = RAMP.SCAD_s
  Runtime = RAMP_Strong.time
  return(list(models, Runtime))
}

RMCPW.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Weak.time <- system.time(RAMP.MCP_w <- RAMP(as.matrix(train.DF[,-1]), as.numeric(train.DF$Y), family = "binomial",
                                                   penalty = "MCP", gamma = NULL, inter = TRUE, hier = "Weak",
                                                   eps = 1e-10, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                   inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                   ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RMCPW"]] = RAMP.MCP_w
  Runtime = RAMP_Weak.time
  return(list(models, Runtime))
}

RMCPS.func <- function(train.DF) {
  models = list()
  Runtime = list()
  RAMP_Strong.time <- system.time(RAMP.MCP_s <- RAMP(as.matrix(train.DF[,-1]), as.numeric(train.DF$Y), family = "binomial",
                                                     penalty = "MCP", gamma = NULL, inter = TRUE, hier = "Strong",
                                                     eps = 1e-10, tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[,-1])),
                                                     inter.penalty.factor = 1, max.iter = 100, n.lambda = 100,
                                                     ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  models[["RMCPS"]] = RAMP.MCP_s
  Runtime = RAMP_Strong.time
  return(list(models, Runtime))
}

library(iRF)
iRF.func <- function(train.DF) {
  models.iRF = list()
  Runtime = list()
  
  iRF.Time <- system.time({
    Mod.iRF <- iRF(as.matrix(train.DF[,-1]), factor(train.DF$Y), xtest = NULL, ytest = NULL,
                   n.iter = 5,
                   ntree = 500,
                   n.core = 5,
                   mtry.select.prob = rep(1/ncol(train.DF[,-1]), ncol(train.DF[,-1])),
                   keep.impvar.quantile = NULL,
                   interactions.return = c(5),
                   wt.pred.accuracy = FALSE,
                   cutoff.unimp.feature = 0,
                   rit.param = list(depth = 20, ntree = 100, nchild = 2, class.id = 1, class.cut = NULL),
                   varnames.grp = NULL,
                   n.bootstrap = 30,
                   bootstrap.forest = TRUE,
                   verbose = TRUE
    )
  })
  
  models.iRF[["iRF"]] = Mod.iRF
  Runtime = iRF.Time
  return(list(models.iRF, Runtime))
}

RealImb.func <- function(n, nsim){
  #To save the simulation results
  CM.LASSO <- list(NULL)
  CM.SCAD <- list(NULL)
  CM.MCP <- list(NULL)
  CM.LW <- list(NULL)
  CM.LS <- list(NULL)
  CM.SW <- list(NULL)
  CM.SS <- list(NULL)
  CM.MCPW <- list(NULL)
  CM.MCPS <- list(NULL)
  MCC.ALL <- list(NULL)
  CM.iRF <- list(NULL)
  MI.ALL <- list(NULL)
  IInd.ALL <- list(NULL)
  MC.ALL <- list(NULL)
  IntCoef.ALL <- list(NULL)
  
  iter <- seq(1,nsim)
  simClass.f <- function(iter){
    # generating data and splitting to train/test
    train.id = sample(n, floor(0.60*n), replace = FALSE, prob = NULL)
    train.DF = DF.Imb[train.id, ]
    test.DF = DF.Imb[-train.id, ]
    
    LASSO.res.Bin = LASSO.func(train.DF)
    SCAD.res.Bin = SCAD.func(train.DF)
    MCP.res.Bin = MCP.func(train.DF)
    RAMP.LW.Bin = RLW.func(train.DF)
    RAMP.LS.Bin = RLS.func(train.DF)
    RAMP.SW.Bin = RSW.func(train.DF)
    RAMP.SS.Bin = RSS.func(train.DF)
    RAMP.MW.Bin = RMCPW.func(train.DF)
    RAMP.MS.Bin = RMCPS.func(train.DF)
    iRF.Bin = iRF.func(train.DF)
    
    ## Evaluating Interaction Selection Performance by Computing Beta Sensitivity and Specificity ##
    
    MEI.RLW = RAMP.LW.Bin[[1]]$RLW$mainInd # Index for the selected Main Effects.
    IInd.RLW = RAMP.LW.Bin[[1]]$RLW$interInd # Index for the selected Interaction Effects
    MECoef.RLW = RAMP.LW.Bin[[1]]$RLW$beta.m # Coefficients for the selected Main Effects
    IntCoef.RLW = RAMP.LW.Bin[[1]]$RLW$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RLW) = IInd.RLW
    
    MEI.RLS = RAMP.LS.Bin[[1]]$RLS$mainInd # Index for the selected Main Effects.
    IInd.RLS = RAMP.LS.Bin[[1]]$RLS$interInd # Index for the selected Interaction Effects
    MECoef.RLS = RAMP.LS.Bin[[1]]$RLS$beta.m # Coefficients for the selected Main Effects
    IntCoef.RLS = RAMP.LS.Bin[[1]]$RLS$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RLS) = IInd.RLS
    
    MEI.RSW = RAMP.SW.Bin[[1]]$RSW$mainInd # Index for the selected Main Effects.
    IInd.RSW = RAMP.SW.Bin[[1]]$RSW$interInd # Index for the selected Interaction Effects
    MECoef.RSW = RAMP.SW.Bin[[1]]$RSW$beta.m # Coefficients for the selected Main Effects
    IntCoef.RSW = RAMP.SW.Bin[[1]]$RSW$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RSW) = IInd.RSW
    
    MEI.RSS = RAMP.SS.Bin[[1]]$RSS$mainInd # Index for the selected Main Effects.
    IInd.RSS = RAMP.SS.Bin[[1]]$RSS$interInd # Index for the selected Interaction Effects
    MECoef.RSS = RAMP.SS.Bin[[1]]$RSS$beta.m # Coefficients for the selected Main Effects
    IntCoef.RSS = RAMP.SS.Bin[[1]]$RSS$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RSS) = IInd.RSS
    
    MEI.RMCPW = RAMP.MW.Bin[[1]]$RMCPW$mainInd # Index for the selected Main Effects.
    IInd.RMCPW = RAMP.MW.Bin[[1]]$RMCPW$interInd # Index for the selected Interaction Effects
    MECoef.RMCPW = RAMP.MW.Bin[[1]]$RMCPW$beta.m # Coefficients for the selected Main Effects
    IntCoef.RMCPW = RAMP.MW.Bin[[1]]$RMCPW$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RMCPW) = IInd.RMCPW
    
    MEI.RMCPS = RAMP.MS.Bin[[1]]$RMCPS$mainInd # Index for the selected Main Effects.
    IInd.RMCPS = RAMP.MS.Bin[[1]]$RMCPS$interInd # Index for the selected Interaction Effects
    MECoef.RMCPS = RAMP.MS.Bin[[1]]$RMCPS$beta.m # Coefficients for the selected Main Effects
    IntCoef.RMCPS = RAMP.MS.Bin[[1]]$RMCPS$beta.i # Coefficients for the selected Interaction Effects
    names(IntCoef.RMCPS) = IInd.RMCPS
    
    IntCoef.iRF = iRF.Bin[[1]][['iRF']]$interaction[[5]]
    IntCoef.iRF.Sig = IntCoef.iRF[IntCoef.iRF>=0.7]
    
    # Evaluating Interaction Selection Performance by Computing Confusion Matrix to report Accuracy, F1-score, Sensitivity and Specificity
    pred.LASSO = predict(LASSO.res.Bin[[1]]$lasso, as.matrix(test.DF[,-1]), type = "class", s = "lambda.min")
    pred.SCAD = predict(SCAD.res.Bin[[1]]$scad, as.matrix(test.DF[,-1]), type = "class")
    pred.MCP = predict(MCP.res.Bin[[1]]$mcp, as.matrix(test.DF[,-1]), type = "class")
    pred.RLW = predict(RAMP.LW.Bin[[1]][["RLW"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RLS <- predict(RAMP.LS.Bin[[1]][["RLS"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RSW <- predict(RAMP.SW.Bin[[1]][["RSW"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RSS <- predict(RAMP.SS.Bin[[1]][["RSS"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RMW <- predict(RAMP.MW.Bin[[1]][["RMCPW"]], as.matrix(test.DF[,-1]), type = "class")
    pred.RMS <- predict(RAMP.MS.Bin[[1]][["RMCPS"]], as.matrix(test.DF[,-1]), type = "class")
    pred.iRF <- predict(iRF.Bin[[1]][["iRF"]]$rf.list, as.matrix(test.DF[,-1]), type = "class")
    
    yhat.RLW <- as.integer(pred.RLW)
    yhat.RLS <- as.integer(pred.RLS)
    yhat.RSW <- as.integer(pred.RSW)
    yhat.RSS <- as.integer(pred.RSS)
    yhat.RMW <- as.integer(pred.RMW)
    yhat.RMS <- as.integer(pred.RMS)
    
    CM.BinLASSO <- confusionMatrix(factor(pred.LASSO), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.LASSO = round(mccr(test.DF$Y, factor(pred.LASSO)), 4)
    CM.BinSCAD <- confusionMatrix(factor(pred.SCAD), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.SCAD = round(mccr(test.DF$Y, factor(pred.SCAD)), 4)
    CM.BinMCP <- confusionMatrix(factor(pred.MCP), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.MCP = round(mccr(test.DF$Y, factor(pred.MCP)), 4)
    CM.BinRLW <- confusionMatrix(factor(yhat.RLW), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RLW = round(mccr(test.DF$Y, factor(yhat.RLW)), 4)
    CM.BinRLS <- confusionMatrix(factor(yhat.RLS), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RLS = round(mccr(test.DF$Y, factor(yhat.RLS)), 4)
    CM.BinRSW <- confusionMatrix(factor(yhat.RSW), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RSW = round(mccr(test.DF$Y, factor(yhat.RSW)), 4)
    CM.BinRSS <- confusionMatrix(factor(yhat.RSS), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RSS = round(mccr(test.DF$Y, factor(yhat.RSS)), 4)
    CM.BinRMW <- confusionMatrix(factor(yhat.RMW), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RMW = round(mccr(test.DF$Y, factor(yhat.RMW)), 4)
    CM.BinRMS <- confusionMatrix(factor(yhat.RMS), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.RMS = round(mccr(test.DF$Y, factor(yhat.RMS)), 4)
    
    CM.BiniRF1 <- confusionMatrix(factor(pred.iRF[[1]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF1 = round(mccr(test.DF$Y, pred.iRF[[1]]), 4)
    CM.BiniRF2 <- confusionMatrix(factor(pred.iRF[[2]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF2 = round(mccr(test.DF$Y, pred.iRF[[2]]), 4)
    CM.BiniRF3 <- confusionMatrix(factor(pred.iRF[[3]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF3 = round(mccr(test.DF$Y, pred.iRF[[3]]), 4)
    CM.BiniRF4 <- confusionMatrix(factor(pred.iRF[[4]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF4 = round(mccr(test.DF$Y, pred.iRF[[4]]), 4)
    CM.BiniRF5 <- confusionMatrix(factor(pred.iRF[[5]]), factor(test.DF$Y), positive = "1", mode = "everything")
    MCC.iRF5 = round(mccr(test.DF$Y, pred.iRF[[5]]), 4)
    
    
    Metrics <- list(MEI.RLW, MEI.RLS, MEI.RSW, MEI.RSS, MEI.RMCPW, MEI.RMCPS, IInd.RLW, IInd.RLS, IInd.RSW, IInd.RSS, IInd.RMCPW, IInd.RMCPS, 
                    MECoef.RLW, MECoef.RLS, MECoef.RSW, MECoef.RSS, MECoef.RMCPW, MECoef.RMCPS, IntCoef.RLW, IntCoef.RLS, IntCoef.RSW, IntCoef.RSS,
                    IntCoef.RMCPW, IntCoef.RMCPS, IntCoef.iRF.Sig, CM.BinLASSO, MCC.LASSO, CM.BinSCAD, MCC.SCAD, CM.BinMCP, MCC.MCP, CM.BinRLW, MCC.RLW,
                    CM.BinRLS, MCC.RLS, CM.BinRSW, MCC.RSW, CM.BinRSS, MCC.RSS, CM.BinRMW, MCC.RMW, CM.BinRMS, MCC.RMS, CM.BiniRF1, MCC.iRF1, CM.BiniRF2,
                    MCC.iRF2, CM.BiniRF3, MCC.iRF3, CM.BiniRF4, MCC.iRF4, CM.BiniRF5, MCC.iRF5)
    
    return(Metrics)
    
  }
  
  numCores <- detectCores()
  results <- mclapply(iter, simClass.f, mc.cores = 6)
  
  #Collect values of estimates
  MEI_RLW = (do.call(cbind, results)[1,])
  MEI_RLS = (do.call(cbind, results)[2,])
  MEI_RSW = (do.call(cbind, results)[3,])
  MEI_RSS = (do.call(cbind, results)[4,])
  MEI_RMCPW = (do.call(cbind, results)[5,])
  MEI_RMCPS = (do.call(cbind, results)[6,])
  
  IInd_RLW = (do.call(cbind, results)[7,])
  IInd_RLS = (do.call(cbind, results)[8,])
  IInd_RSW = (do.call(cbind, results)[9,])
  IInd_RSS = (do.call(cbind, results)[10,])
  IInd_RMCPW = (do.call(cbind, results)[11,])
  IInd_RMCPS = (do.call(cbind, results)[12,])
  
  MECoef_RLW = (do.call(cbind, results)[13,])
  MECoef_RLS = (do.call(cbind, results)[14,])
  MECoef_RSW = (do.call(cbind, results)[15,])
  MECoef_RSS = (do.call(cbind, results)[16,])
  MECoef_RMCPW = (do.call(cbind, results)[17,])
  MECoef_RMCPS = (do.call(cbind, results)[18,])
  
  IntCoef_RLW = (do.call(cbind, results)[19,])
  IntCoef_RLS = (do.call(cbind, results)[20,])
  IntCoef_RSW = (do.call(cbind, results)[21,])
  IntCoef_RSS = (do.call(cbind, results)[22,])
  IntCoef_RMCPW = (do.call(cbind, results)[23,])
  IntCoef_RMCPS = (do.call(cbind, results)[24,])
  IntCoef_iRF.Sig = (do.call(cbind, results)[25,])
  
  CM_LASSO = (do.call(cbind, results)[26,])
  MCC_LASSO = (do.call(cbind, results)[27,])
  CM_SCAD = (do.call(cbind, results)[28,])
  MCC_SCAD = (do.call(cbind, results)[29,])
  CM_MCP = (do.call(cbind, results)[30,])
  MCC_MCP = (do.call(cbind, results)[31,])
  CM_RLW = (do.call(cbind, results)[32,])
  MCC_RLW = (do.call(cbind, results)[33,])
  CM_RLS = (do.call(cbind, results)[34,])
  MCC_RLS = (do.call(cbind, results)[35,])
  CM_RSW = (do.call(cbind, results)[36,])
  MCC_RSW = (do.call(cbind, results)[37,])
  CM_RSS = (do.call(cbind, results)[38,])
  MCC_RSS = (do.call(cbind, results)[39,])
  CM_RMCPW = (do.call(cbind, results)[40,])
  MCC_RMW = (do.call(cbind, results)[41,])
  CM_RMCPS = (do.call(cbind, results)[42,])
  MCC_RMS = (do.call(cbind, results)[43,])
  
  CM_iRF1 = (do.call(cbind, results)[44,])
  MCC_iRF1 = (do.call(cbind, results)[45,])
  CM_iRF2 = (do.call(cbind, results)[46,])
  MCC_iRF2 = (do.call(cbind, results)[47,])
  CM_iRF3 = (do.call(cbind, results)[48,])
  MCC_iRF3 = (do.call(cbind, results)[49,])
  CM_iRF4 = (do.call(cbind, results)[50,])
  MCC_iRF4 = (do.call(cbind, results)[51,])
  CM_iRF5 = (do.call(cbind, results)[52,])
  MCC_iRF5 = (do.call(cbind, results)[53,])
  
  CM.LASSO <- list(CM_LASSO)
  CM.SCAD <- list(CM_SCAD)
  CM.MCP <- list(CM_MCP)
  CM.LW = list(CM_RLW)
  CM.LS = list(CM_RLS)
  CM.SW = list(CM_RSW)
  CM.SS = list(CM_RSS)
  CM.MCPW = list(CM_RMCPW)
  CM.MCPS = list(CM_RMCPS)
  CM.iRF = list(CM_iRF1, CM_iRF2, CM_iRF3, CM_iRF4, CM_iRF5)
  
  MCC.ALL = list(MCC_LASSO, MCC_SCAD, MCC_MCP, MCC_RLW, MCC_RLS, MCC_RSW, MCC_RSS, MCC_RMW, MCC_RMS, MCC_iRF1, MCC_iRF2, MCC_iRF3, MCC_iRF4, MCC_iRF5)
  MI.ALL = list(MEI_RLW, MEI_RLS, MEI_RSW, MEI_RSS, MEI_RMCPW, MEI_RMCPS)
  IInd.ALL = list(IInd_RLW, IInd_RLS, IInd_RSW, IInd_RSS, IInd_RMCPW, IInd_RMCPS) 
  MC.ALL = list(MECoef_RLW, MECoef_RLS, MECoef_RSW, MECoef_RSS, MECoef_RMCPW, MECoef_RMCPS)
  IntCoef.ALL = list(IntCoef_RLW, IntCoef_RLS, IntCoef_RSW, IntCoef_RSS, IntCoef_RMCPW, IntCoef_RMCPS, IntCoef_iRF.Sig)
  
  result.final1 <- list(CM.LASSO, CM.SCAD, CM.MCP, CM.LW, CM.LS, CM.SW, CM.SS, CM.MCPW, CM.MCPS, CM.iRF, MCC.ALL)
  
  
  result.final2 <- list(MI.ALL, IInd.ALL, MC.ALL, IntCoef.ALL)
  
  return(list(result.final1, result.final2))
}

set.seed(1788)

n <- nrow(DF.Imb)
nsim <- 100
system.time(RealImb <- RealImb.func(n,nsim))


###########################################################################################
######## Evaluating Performance by Computing Accuracy, Sensitivity, Specificity ###########
######## F1-Score, Balanced Accuracy and Matthew's Correlation Coefficients ###############
###########################################################################################
LAS.Acc <- matrix(NA, nrow = nsim, ncol = 1)
LAS.Sens <- matrix(NA, nrow = nsim, ncol = 1)
LAS.Spec <- matrix(NA, nrow = nsim, ncol = 1)
LAS.F1 <- matrix(NA, nrow = nsim, ncol = 1)
LAS.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

SCAD.Acc <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.Sens <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.Spec <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.F1 <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

MCP.Acc <- matrix(NA, nrow = nsim, ncol = 1)
MCP.Sens <- matrix(NA, nrow = nsim, ncol = 1)
MCP.Spec <- matrix(NA, nrow = nsim, ncol = 1)
MCP.F1 <- matrix(NA, nrow = nsim, ncol = 1)
MCP.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RLW.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RLW.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RLW.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RLW.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RLW.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RLS.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RLS.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RLS.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RLS.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RLS.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RSW.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RSW.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RSW.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RSW.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RSW.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RSS.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RSS.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RSS.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RSS.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RSS.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RMCPW.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

RMCPS.Acc <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.Sens <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.Spec <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.F1 <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF1.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF2.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF3.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF4.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

iRF5.Acc <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.Sens <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.Spec <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.F1 <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.BalAcc <- matrix(NA, nrow = nsim, ncol = 1)

LAS.MCC <- matrix(NA, nrow = nsim, ncol = 1)
SCAD.MCC <- matrix(NA, nrow = nsim, ncol = 1)
MCP.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RLW.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RLS.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RSW.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RSS.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RMCPW.MCC <- matrix(NA, nrow = nsim, ncol = 1)
RMCPS.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF1.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF2.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF3.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF4.MCC <- matrix(NA, nrow = nsim, ncol = 1)
iRF5.MCC <- matrix(NA, nrow = nsim, ncol = 1)

for (i in 1:nsim) {
  LAS.Acc[i,] = RealBal[[1]][[1]][[1]][[i]]$overall[1]
  LAS.Sens[i,] = RealBal[[1]][[1]][[1]][[i]]$byClass[1]
  LAS.Spec[i,] = RealBal[[1]][[1]][[1]][[i]]$byClass[2]
  LAS.F1[i,] = RealBal[[1]][[1]][[1]][[i]]$byClass[7]
  LAS.BalAcc[i,] = RealBal[[1]][[1]][[1]][[i]]$byClass[11]
  
  SCAD.Acc[i,] = RealBal[[1]][[2]][[1]][[i]]$overall[1]
  SCAD.Sens[i,] = RealBal[[1]][[2]][[1]][[i]]$byClass[1]
  SCAD.Spec[i,] = RealBal[[1]][[2]][[1]][[i]]$byClass[2]
  SCAD.F1[i,] = RealBal[[1]][[2]][[1]][[i]]$byClass[7]
  SCAD.BalAcc[i,] = RealBal[[1]][[2]][[1]][[i]]$byClass[11]
  
  MCP.Acc[i,] = RealBal[[1]][[3]][[1]][[i]]$overall[1]
  MCP.Sens[i,] = RealBal[[1]][[3]][[1]][[i]]$byClass[1]
  MCP.Spec[i,] = RealBal[[1]][[3]][[1]][[i]]$byClass[2]
  MCP.F1[i,] = RealBal[[1]][[3]][[1]][[i]]$byClass[7]
  MCP.BalAcc[i,] = RealBal[[1]][[3]][[1]][[i]]$byClass[11]
  
  RLW.Acc[i,] = RealBal[[1]][[4]][[1]][[i]]$overall[1]
  RLW.Sens[i,] = RealBal[[1]][[4]][[1]][[i]]$byClass[1]
  RLW.Spec[i,] = RealBal[[1]][[4]][[1]][[i]]$byClass[2]
  RLW.F1[i,] = RealBal[[1]][[4]][[1]][[i]]$byClass[7]
  RLW.BalAcc[i,] = RealBal[[1]][[4]][[1]][[i]]$byClass[11]
  
  RLS.Acc[i,] = RealBal[[1]][[5]][[1]][[i]]$overall[1]
  RLS.Sens[i,] = RealBal[[1]][[5]][[1]][[i]]$byClass[1]
  RLS.Spec[i,] = RealBal[[1]][[5]][[1]][[i]]$byClass[2]
  RLS.F1[i,] = RealBal[[1]][[5]][[1]][[i]]$byClass[7]
  RLS.BalAcc[i,] = RealBal[[1]][[5]][[1]][[i]]$byClass[11]
  
  RSW.Acc[i,] = RealBal[[1]][[6]][[1]][[i]]$overall[1]
  RSW.Sens[i,] = RealBal[[1]][[6]][[1]][[i]]$byClass[1]
  RSW.Spec[i,] = RealBal[[1]][[6]][[1]][[i]]$byClass[2]
  RSW.F1[i,] = RealBal[[1]][[6]][[1]][[i]]$byClass[7]
  RSW.BalAcc[i,] = RealBal[[1]][[6]][[1]][[i]]$byClass[11]
  
  RSS.Acc[i,] = RealBal[[1]][[7]][[1]][[i]]$overall[1]
  RSS.Sens[i,] = RealBal[[1]][[7]][[1]][[i]]$byClass[1]
  RSS.Spec[i,] = RealBal[[1]][[7]][[1]][[i]]$byClass[2]
  RSS.F1[i,] = RealBal[[1]][[7]][[1]][[i]]$byClass[7]
  RSS.BalAcc[i,] = RealBal[[1]][[7]][[1]][[i]]$byClass[11]
  
  RMCPW.Acc[i,] = RealBal[[1]][[8]][[1]][[i]]$overall[1]
  RMCPW.Sens[i,] = RealBal[[1]][[8]][[1]][[i]]$byClass[1]
  RMCPW.Spec[i,] = RealBal[[1]][[8]][[1]][[i]]$byClass[2]
  RMCPW.F1[i,] = RealBal[[1]][[8]][[1]][[i]]$byClass[7]
  RMCPW.BalAcc[i,] = RealBal[[1]][[8]][[1]][[i]]$byClass[11]
  
  RMCPS.Acc[i,] = RealBal[[1]][[9]][[1]][[i]]$overall[1]
  RMCPS.Sens[i,] = RealBal[[1]][[9]][[1]][[i]]$byClass[1]
  RMCPS.Spec[i,] = RealBal[[1]][[9]][[1]][[i]]$byClass[2]
  RMCPS.F1[i,] = RealBal[[1]][[9]][[1]][[i]]$byClass[7]
  RMCPS.BalAcc[i,] = RealBal[[1]][[9]][[1]][[i]]$byClass[11]
  
  iRF1.Acc[i,] = RealBal[[1]][[10]][[1]][[i]]$overall[1]
  iRF1.Sens[i,] = RealBal[[1]][[10]][[1]][[i]]$byClass[1]
  iRF1.Spec[i,] = RealBal[[1]][[10]][[1]][[i]]$byClass[2]
  iRF1.F1[i,] = RealBal[[1]][[10]][[1]][[i]]$byClass[7]
  iRF1.BalAcc[i,] = RealBal[[1]][[10]][[1]][[i]]$byClass[11]
  
  iRF2.Acc[i,] = RealBal[[1]][[10]][[2]][[i]]$overall[1]
  iRF2.Sens[i,] = RealBal[[1]][[10]][[2]][[i]]$byClass[1]
  iRF2.Spec[i,] = RealBal[[1]][[10]][[2]][[i]]$byClass[2]
  iRF2.F1[i,] = RealBal[[1]][[10]][[2]][[i]]$byClass[7]
  iRF2.BalAcc[i,] = RealBal[[1]][[10]][[2]][[i]]$byClass[11]
  
  iRF3.Acc[i,] = RealBal[[1]][[10]][[3]][[i]]$overall[1]
  iRF3.Sens[i,] = RealBal[[1]][[10]][[3]][[i]]$byClass[1]
  iRF3.Spec[i,] = RealBal[[1]][[10]][[3]][[i]]$byClass[2]
  iRF3.F1[i,] = RealBal[[1]][[10]][[3]][[i]]$byClass[7]
  iRF3.BalAcc[i,] = RealBal[[1]][[10]][[3]][[i]]$byClass[11]
  
  iRF4.Acc[i,] = RealBal[[1]][[10]][[4]][[i]]$overall[1]
  iRF4.Sens[i,] = RealBal[[1]][[10]][[4]][[i]]$byClass[1]
  iRF4.Spec[i,] = RealBal[[1]][[10]][[4]][[i]]$byClass[2]
  iRF4.F1[i,] = RealBal[[1]][[10]][[4]][[i]]$byClass[7]
  iRF4.BalAcc[i,] = RealBal[[1]][[10]][[4]][[i]]$byClass[11]
  
  iRF5.Acc[i,] = RealBal[[1]][[10]][[5]][[i]]$overall[1]
  iRF5.Sens[i,] = RealBal[[1]][[10]][[5]][[i]]$byClass[1]
  iRF5.Spec[i,] = RealBal[[1]][[10]][[5]][[i]]$byClass[2]
  iRF5.F1[i,] = RealBal[[1]][[10]][[5]][[i]]$byClass[7]
  iRF5.BalAcc[i,] = RealBal[[1]][[10]][[5]][[i]]$byClass[11]
  
  LAS.MCC[i,] = RealBal[[1]][[11]][[1]][[i]]
  SCAD.MCC[i,] = RealBal[[1]][[11]][[2]][[i]]
  MCP.MCC[i,] = RealBal[[1]][[11]][[3]][[i]]
  RLW.MCC[i,] = RealBal[[1]][[11]][[4]][[i]]
  RLS.MCC[i,] = RealBal[[1]][[11]][[5]][[i]]
  RSW.MCC[i,] = RealBal[[1]][[11]][[6]][[i]]
  RSS.MCC[i,] = RealBal[[1]][[11]][[7]][[i]]
  RMCPW.MCC[i,] = RealBal[[1]][[11]][[8]][[i]]
  RMCPS.MCC[i,] = RealBal[[1]][[11]][[9]][[i]]
  iRF1.MCC[i,] = RealBal[[1]][[11]][[10]][[i]]
  iRF2.MCC[i,] = RealBal[[1]][[11]][[11]][[i]]
  iRF3.MCC[i,] = RealBal[[1]][[11]][[12]][[i]]
  iRF4.MCC[i,] = RealBal[[1]][[11]][[13]][[i]]
  iRF5.MCC[i,] = RealBal[[1]][[11]][[14]][[i]]
  
  Pred.Result = cbind(LAS.Acc, LAS.Sens, LAS.Spec, LAS.F1, LAS.BalAcc, LAS.MCC,
                      SCAD.Acc, SCAD.Sens, SCAD.Spec, SCAD.F1, SCAD.BalAcc, SCAD.MCC,
                      MCP.Acc, MCP.Sens, MCP.Spec, MCP.F1, MCP.BalAcc, MCP.MCC,
                      RLW.Acc, RLW.Sens, RLW.Spec, RLW.F1, RLW.BalAcc, RLW.MCC,
                      RLS.Acc, RLS.Sens, RLS.Spec, RLS.F1, RLS.BalAcc, RLS.MCC,
                      RSW.Acc, RSW.Sens, RSW.Spec, RSW.F1, RSW.BalAcc, RSW.MCC,
                      RSS.Acc, RSS.Sens, RSS.Spec, RSS.F1, RSS.BalAcc, RSS.MCC,
                      RMCPW.Acc, RMCPW.Sens, RMCPW.Spec, RMCPW.F1, RMCPW.BalAcc, RMCPW.MCC,
                      RMCPS.Acc, RMCPS.Sens, RMCPS.Spec, RMCPS.F1, RMCPS.BalAcc, RMCPS.MCC,
                      iRF1.Acc, iRF1.Sens, iRF1.Spec, iRF1.F1, iRF1.BalAcc, iRF1.MCC,
                      iRF2.Acc, iRF2.Sens, iRF2.Spec, iRF2.F1, iRF2.BalAcc, iRF2.MCC,
                      iRF3.Acc, iRF3.Sens, iRF3.Spec, iRF3.F1, iRF3.BalAcc, iRF3.MCC,
                      iRF4.Acc, iRF4.Sens, iRF4.Spec, iRF4.F1, iRF4.BalAcc, iRF4.MCC,
                      iRF5.Acc, iRF5.Sens, iRF5.Spec, iRF5.F1, iRF5.BalAcc, iRF5.MCC)
}

colnames(Pred.Result) = c("LAS.Acc", "LAS.Sens", "LAS.Spec", "LAS.F1", "LAS.BalAcc", "LAS.MCC",
                          "SCAD.Acc", "SCAD.Sens", "SCAD.Spec", "SCAD.F1", "SCAD.BalAcc", "SCAD.MCC",
                          "MCP.Acc", "MCP.Sens", "MCP.Spec", "MCP.F1", "MCP.BalAcc", "MCP.MCC",
                          "RLW.Acc", "RLW.Sens", "RLW.Spec", "RLW.F1", "RLW.BalAcc", "RLW.MCC",
                          "RLS.Acc", "RLS.Sens", "RLS.Spec", "RLS.F1", "RLS.BalAcc", "RLS.MCC",
                          "RSW.Acc", "RSW.Sens", "RSW.Spec", "RSW.F1", "RSW.BalAcc", "RSW.MCC",
                          "RSS.Acc", "RSS.Sens", "RSS.Spec", "RSS.F1", "RSS.BalAcc", "RSS.MCC",
                          "RMCPW.Acc", "RMCPW.Sens", "RMCPW.Spec", "RMCPW.F1", "RMCPW.BalAcc", "RMCPW.MCC",
                          "RMCPS.Acc", "RMCPS.Sens", "RMCPS.Spec", "RMCPS.F1", "RMCPS.BalAcc", "RMCPS.MCC",
                          "iRF1.Acc", "iRF1.Sens", "iRF1.Spec", "iRF1.F1", "iRF1.BalAcc", "iRF1.MCC",
                          "iRF2.Acc", "iRF2.Sens", "iRF2.Spec", "iRF2.F1", "iRF2.BalAcc", "iRF2.MCC",
                          "iRF3.Acc", "iRF3.Sens", "iRF3.Spec", "iRF3.F1", "iRF3.BalAcc", "iRF3.MCC",
                          "iRF4.Acc", "iRF4.Sens", "iRF4.Spec", "iRF4.F1", "iRF4.BalAcc", "iRF4.MCC",
                          "iRF5.Acc", "iRF5.Sens", "iRF5.Spec", "iRF5.F1", "iRF5.BalAcc", "iRF5.MCC")
Pred.Real = Pred.Result

PredResult <- as.data.frame(Pred.Real)

PredMetrics <- PredResult %>% 
  dplyr::summarise_each(funs(mean, sd, SE = sd(.)/sqrt(n())))

ClassReal <- PredMetrics %>% 
  tidyr::gather(key = Pred, value = value) 
ClassReal.sep <- separate(ClassReal,
                          col = Pred,
                          sep = "_",
                          into = c("Model", "Metrics"))

Metrics.ClassReal <- ClassReal.sep %>% 
  spread(key = Metrics, value = value)


