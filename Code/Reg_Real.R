###########################################################################

rm(list = ls())

library(tidyverse)
library(MASS) # for mvrnorm
library(mvnfast) # for dmvn
library(Matrix)  
library(caret)
library(glmnet)
library(ncvreg)
library(RAMP)
library(iRF).         # For this study we used iRF 2.0.0 (`devtools::install_github("karlkumbier/iRF2.0")`)
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(Boruta)       # Selection of important features for RF
library(parallel)

(core.num <- detectCores())

setwd("Path")

DF.Gen <- readRDS("bcTCGA.rds")
DF.Gen[["X"]] <- scale(DF.Gen[["X"]], center = TRUE, scale = TRUE) # scale is generic function whose default method centers and/or scales the columns of a numeric matrix.If center is TRUE then centering is done by subtracting the column means (omitting NAs) of x from their corresponding columns, and if center is FALSE, no centering is done.
DF.Gen <- as.data.frame(cbind(DF.Gen[["y"]], DF.Gen[["X"]]))

colnames(DF.Gen)[1] <- "Y"

####################################################################
########################## Prediction Metric #######################
####################################################################

mse.fun <- function(x,y){mean((x-y)^2)}

####################################################################
############### Interaction Selection Algorithms ###################
####################################################################
LASSO.func <- function(train.DF, test.DF) {
  LASSO.time <- system.time(
    {
      LASSO.Mod <- cv.glmnet(x = as.matrix(train.DF[,-1]), y = as.matrix(train.DF$Y),family = "gaussian", alpha = 1)
      LASSO.coef = coef(LASSO.Mod, s="lambda.min")
      LASSO.pred.tr = predict(LASSO.Mod, as.matrix(train.DF[,-1]), s = "lambda.min")
      LASSO.pred.ts = predict(LASSO.Mod, as.matrix(test.DF[,-1]), s = "lambda.min")
      LASSO.MSE.tr = mse.fun(LASSO.pred.tr, train.DF$Y)
      LASSO.MSE.ts = mse.fun(LASSO.pred.ts, test.DF$Y)
      rm(LASSO.Mod)
    }
  )
  return(list(LASSO.time, LASSO.coef, LASSO.MSE.tr, LASSO.MSE.ts))
}

NCP.func <- function(train.DF, test.DF, method) {
  
  NCP.time <- system.time(
    {
      NCP.Mod <- cv.ncvreg(X = as.matrix(train.DF[,-1]), y = as.matrix(train.DF$Y), family = "gaussian", penalty = method)
      NCP.coef <- coef(NCP.Mod)
      NCP.pred.tr = predict(NCP.Mod, as.matrix(train.DF[,-1]))
      NCP.pred.ts = predict(NCP.Mod, as.matrix(test.DF[,-1]))
      NCP.MSE.tr = mse.fun(NCP.pred.tr, train.DF$Y)
      NCP.MSE.ts = mse.fun(NCP.pred.ts, test.DF$Y)
      rm(NCP.Mod)
    }
  )  
  return(list(NCP.time, NCP.coef, NCP.MSE.tr, NCP.MSE.ts))
}

###################################################

# RAMP ALGORITHM for three penalty functions and two types of interactions
RAMP.func <- function(train.DF, test.DF, penalty.val, hier.val){
  RAMP.Time <- system.time(
    temp.tr <- RAMP(X = as.matrix(train.DF[, -1]), y = as.matrix(train.DF[, 1]), family = "gaussian",
                    penalty = penalty.val, gamma = NULL, inter = TRUE, hier = hier.val, eps = 1e-15,
                    tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[, -1])), inter.penalty.factor = 1,
                    max.iter = 200, n.lambda = iter, ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  temp.fit <- predict(temp.tr, as.matrix(train.DF[, -1]))
  temp.pred <- predict(temp.tr, as.matrix(test.DF[, -1]))
  mse.tr <- mse.fun(temp.fit, train.DF[,1] )
  mse.ts <- mse.fun(temp.pred, test.DF[,1] )
  
  return(list(RAMP.Time, temp.tr$mainInd, temp.tr$interInd, temp.tr$beta.m, temp.tr$beta.i, mse.tr, mse.ts))
}

###################################################
RF.func <- function(train.DF, test.DF) {
  
  RF.Time <- system.time({
    RF.tr = randomForest::randomForest(
      x         = train.DF[,-1],
      y         = train.DF[,1],
      ntree     = RF.BestGrid$n.trees[1],
      mtry      = RF.BestGrid$m.try[1],
      nodesize  = 5 # Depth of the decision tree
    )
    
    RF.fit <- predict(RF.tr, as.matrix(train.DF[,-1]))
    RF.pred <- predict(RF.tr, as.matrix(test.DF[,-1]))
    RF.mse.tr <- mse.fun(RF.fit, train.DF$Y)
    RF.mse.ts <- mse.fun(RF.pred, test.DF$Y)
    rm(RF.tr)
  }
  )
  return(list(RF.Time, RF.mse.tr, RF.mse.ts))
}



iRF.func <- function(train.DF, test.DF) {
  
  iRF.Time <- system.time(
    {
      iRF.tr <- iRF(x = as.matrix(train.DF[, -1]), y = as.matrix(train.DF[, 1]),
                    n.iter = 5,
                    ntree = 500,
                    n.core = core.num - 2,
                    mtry.select.prob = rep(1/ncol(train.DF[, -1]), ncol(train.DF[, -1])),
                    keep.impvar.quantile = NULL,
                    interactions.return = 5,
                    wt.pred.accuracy = FALSE,
                    select.iter = FALSE,
                    cutoff.unimp.feature = 0,
                    rit.param = list(depth = 20, ntree = iter, nchild = 2, class.id = 1, class.cut = NULL),
                    varnames.grp = NULL,
                    n.bootstrap = 30,
                    bootstrap.forest = TRUE,
                    verbose = TRUE)
      
      iRF.fit <- predict(iRF.tr$rf.list[[5]], as.matrix(train.DF[,-1]))
      iRF.pred <- predict(iRF.tr$rf.list[[5]], as.matrix(test.DF[,-1]))
      iRF.inter <- iRF.tr$interaction[[5]]
      iRF.mse.tr <- mse.fun(iRF.fit, train.DF$Y)
      iRF.mse.ts <- mse.fun(iRF.pred, test.DF$Y)
      rm(iRF.tr)
    }
  )
  return(list(iRF.Time, iRF.inter, iRF.mse.tr, iRF.mse.ts))
}

##################################################################3
sim.func <- function(n){
  
  train.id = sample(n, floor(0.60*n), replace = FALSE, prob = NULL)
  
  train.DF = DF.Gen[train.id, ]
  test.DF = DF.Gen[-train.id, ]
  
  LASSO.res = LASSO.func(train.DF, test.DF)
  SCAD.res = NCP.func(train.DF, test.DF, method="SCAD")
  MCP.res = NCP.func(train.DF, test.DF, method="MCP")
  
  LASSO.W.res  = RAMP.func(train.DF, test.DF, penalty.val="LASSO", hier.val="Weak")  # Main effects only
  LASSO.S.res  = RAMP.func(train.DF, test.DF, penalty.val="LASSO", hier.val="Strong")  # Main effects only
  SCAD.W.res  = RAMP.func(train.DF, test.DF, penalty.val="SCAD", hier.val="Weak")  # Main effects only
  SCAD.S.res  = RAMP.func(train.DF, test.DF, penalty.val="SCAD", hier.val="Strong")  # Main effects only
  MCP.W.res  = RAMP.func(train.DF, test.DF, penalty.val="MCP", hier.val="Weak")  # Main effects only
  MCP.S.res  = RAMP.func(train.DF, test.DF, penalty.val="MCP", hier.val="Strong")  # Main effects only
  
  RF.res = RF.func(train.DF, test.DF)
  iRF.res = iRF.func(train.DF, test.DF)                                        # Main effects only
  
  return(list(LASSO.res, SCAD.res, MCP.res,
              LASSO.W.res, LASSO.S.res, SCAD.W.res, SCAD.S.res, 
              MCP.W.res, MCP.S.res, RF.res, iRF.res))
}


###################################################################################

set.seed(12345)
iter = 100
n <- nrow(DF.Gen)


system.time(
  {
    result2 = do.call(cbind, replicate(iter, sim.func(n=n.val)))
  }
)


sink("Path")
print(result2)
sink()

##########################################################
############### Summary of results #######################
##########################################################

dat <- result2

nc <- ncol(dat)

K <- 12*iter

#################################################################
# Step 2. Define related functions

# Evaluation of variable selection
beta.eval.func <- function(b.true, b.est){
  b_sens = sum(b.est[which(b.true!=0)]!=0)/length(b.true[b.true!=0])
  b_spec = sum(b.est[which(b.true==0)]==0)/length(b.true[b.true==0])
  b_fp = sum(b.est[which(b.true==0)]!=0)/length(b.true[b.true==0])
  b_fn = sum(b.est[which(b.true!=0)]==0)/length(b.true[b.true!=0])
  l2 = sqrt(sum(b.true-b.est)^2)
  
  return(c(b_sens, b_spec, b_fp, b_fn, l2))
}

# Constructing the parameter space from RAMP results
b.est.RAMP <- function(beta, dat, meth.pos, iter){
  
  temp.beta <- rep(0,length(beta)); beta.est.RAMP <- matrix(rep(0,iter*length(beta)), ncol=iter)
  names(temp.beta) <- rownames(dat[[2,1]])[-1]
  temp.name <- sapply(1:iter, function(x) c(names(dat[4, 1:1200 %% 12 == meth.pos][[x]]), dat[3, 1:1200 %% 12 == meth.pos][[x]]))
  temp <- sapply(1:iter, function(x) c(unlist(dat[4,1:nc %% 12 == meth.pos][[x]]), unlist(dat[5,1:nc %% 12 == meth.pos][[x]])))
  for (ii in 1:iter) {
    beta.est.RAMP[which(names(temp.beta) %in% temp.name[[ii]]), ii] = temp[[ii]]
    rownames(beta.est.RAMP) = names(temp.beta)
  }
  
  return(beta.est.RAMP)
}

# Constructing the parameter space from HierNet results

b.est.Hier <- function(dat){
  b.est.Hier <- matrix(nrow = length(dat[[2,1]])-1, ncol=iter)
  
  for (i in 1:iter){
    a <- dat[[3, 12*(i-1) + 10 ]]; a[lower.tri(a)]<-NA;  a.na <- na.omit(c(a))
    b.est.Hier[, i] <- c(dat[2, 12*(i-1) + 10][[1]], a.na)
    rownames(b.est.Hier) <- rownames(dat[[2,1]])[-1]
  }
  
  return(b.est.Hier)
}

# Step 3 - Extract and summarize the results

Result.report.fun <- function(dat, mod = 1, p.val = p.val, n.meth = 12, cutoff.val = 0.7, iter = iter){
  
  {
    if (mod==1) beta =  c(0.2, 0.3, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
    else if (mod==2) beta = c(0, 0, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
    else if (mod==3) beta = c(0, 0, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))  # Need to update
    else if (mod==4) beta = c(0.2, 0.3, 0, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
  }
  
  nc = ncol(dat)
  
  method <- c("LASSO", "SCAD", "MCP", "LASSO.W","LASSO.S", "SCAD.W", "SCAD.S", "MCP.W", "MCP.S", "Hier", "RF", "iRF")
  metric <- c("Sens", "Spec", "FP", "FN", "L2")
  
  #### MSE
  
  Sim.time.result <- matrix(nrow = iter, ncol = n.meth)
  MSE.tr.result <- MSE.ts.result <- matrix(nrow=iter, ncol = 2*n.meth)
  Beta.result <- matrix(nrow = iter, ncol = 50)
  
  Sim.time.temp <- sapply(c(1:11,0), function(x) dat[1, ((1:nc) %% n.meth == x)])
  Sim.time.result <- matrix(sapply(Sim.time.temp, "[[", 3), nrow=iter)
  Sim.time <- rbind(apply(Sim.time.result, 2, mean), apply(Sim.time.result, 2, sd))
  colnames(Sim.time) <- method
  rownames(Sim.time) <- c("Mean", "SD")
  
  MSE.tr.result <- data.frame(cbind( sapply(1:3, function(x) dat[3, ((1:nc) %% n.meth == x)]),
                                     sapply(4:9, function(x) dat[6, ((1:nc) %% n.meth == x)]),
                                     sapply(10, function(x) dat[4, ((1:nc) %% n.meth == x)]),
                                     sapply(c(11,0), function(x) dat[3, ((1:nc) %% n.meth == x)])))
  
  MSE.ts.result <- data.frame(cbind( sapply(1:3, function(x) dat[4, ((1:nc) %% n.meth == x)]),
                                     sapply(4:9, function(x) dat[7, ((1:nc) %% n.meth == x)]),
                                     sapply(10, function(x) dat[5, ((1:nc) %% n.meth == x)]),
                                     sapply(c(11,0), function(x) dat[4, ((1:nc) %% n.meth == x)])))
  
  MSE.result <- data.frame(rbind( apply(matrix(unlist(MSE.tr.result), ncol=n.meth), 2, mean),
                                  apply(matrix(unlist(MSE.tr.result), ncol=n.meth),2, sd),
                                  apply(matrix(unlist(MSE.ts.result), ncol=n.meth),2, mean),
                                  apply(matrix(unlist(MSE.ts.result), ncol=n.meth),2, sd)))
  
  colnames(MSE.result) <- method
  MSE.result <- MSE.result %>%
    mutate(Data = c("Train", "Train", "Test", "Test"),
           Stat = c("Mean", "SD", "Mean", "SD")) %>%
    dplyr::select(Data, Stat, everything())
  
  #### Beta evaluation
  
  LASSO_beta <- sapply(1:iter, function(x) beta.eval.func(beta, sapply(dat[2, 1:1200 %% 12 == 1], "[", -1)[,x]))
  SCAD_beta <- sapply(1:iter, function(x) beta.eval.func(beta, sapply(dat[2, 1:1200 %% 12 == 2], "[", -1)[,x]))
  MCP_beta <-  sapply(1:iter, function(x) beta.eval.func(beta, sapply(dat[2, 1:1200 %% 12 == 3], "[", -1)[,x]))
  
  #### RAMP
  LASSO.W.temp <- b.est.RAMP(beta, dat, meth.pos=4, iter)
  LASSO.W_beta <- sapply(1:iter, function(x) beta.eval.func(beta, LASSO.W.temp[,x]))
  LASSO.S.temp <- b.est.RAMP(beta, dat, meth.pos=5, iter)
  LASSO.S_beta <- sapply(1:iter, function(x) beta.eval.func(beta, LASSO.S.temp[,x]))
  SCAD.W.temp <- b.est.RAMP(beta, dat, meth.pos=6, iter)
  SCAD.W_beta <- sapply(1:iter, function(x) beta.eval.func(beta, SCAD.W.temp[,x]))
  SCAD.S.temp <- b.est.RAMP(beta, dat, meth.pos=7, iter)
  SCAD.S_beta <- sapply(1:iter, function(x) beta.eval.func(beta, SCAD.S.temp[,x]))
  MCP.W.temp <- b.est.RAMP(beta, dat, meth.pos=8, iter)
  MCP.W_beta <- sapply(1:iter, function(x) beta.eval.func(beta, MCP.W.temp[,x]))
  MCP.S.temp <- b.est.RAMP(beta, dat, meth.pos=9, iter)
  MCP.S_beta <- sapply(1:iter, function(x) beta.eval.func(beta, MCP.S.temp[,x]))
  
  
  Beta.result <- data.frame(rbind(
    rbind( apply(LASSO_beta,1, mean),apply(LASSO_beta,1, sd)),
    rbind( apply(SCAD_beta,1, mean),apply(SCAD_beta,1, sd)),
    rbind( apply(MCP_beta,1, mean),apply(MCP_beta,1, sd)),
    rbind( apply(LASSO.W_beta,1, mean),apply(LASSO.W_beta,1, sd)),
    rbind( apply(LASSO.S_beta,1, mean),apply(LASSO.S_beta,1, sd)),
    rbind( apply(SCAD.W_beta,1, mean),apply(SCAD.W_beta,1, sd)),
    rbind( apply(SCAD.S_beta,1, mean),apply(SCAD.S_beta,1, sd)),
    rbind( apply(MCP.W_beta,1, mean),apply(MCP.W_beta,1, sd)),
    rbind( apply(MCP.S_beta,1, mean),apply(MCP.S_beta,1, sd)),
    rbind( apply(Hier_beta,1, mean),apply(Hier_beta,1, sd))))
  colnames(Beta.result) <- metric
  Beta.result <- Beta.result %>%
    mutate(Method = rep(method[1:10], rep(2, length(method[1:10]))),
           Stat = rep(c("Mean", "SD"), 10)) %>%
    dplyr::select(Method, Stat, everything())
  
  return(list(Sim.time, MSE.result, Beta.result))
}


Report <- Result.report.fun(dat = result2, mod = mod, p.val = p.val, n.meth = 12, cutoff.val = 0.7, iter = iter)

Report
