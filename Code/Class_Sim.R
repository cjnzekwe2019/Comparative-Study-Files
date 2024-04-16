###########################################################################

rm(list = ls())

library(tidyverse)
library(MASS)         # for mvrnorm
library(mvnfast)      # for dmvn
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
library(mccr)

(core.num <- detectCores())

DF.Gen <- function(n, p, mod){
  # Models from Duroux and Scornet (2018)
  {
    if(mod < 13){
      dependt.Cov = matrix(c(1, 0.3, 0.3, 0.6, 0.6,
                             0.3, 1, 0.3, 0.2, 0.1,
                             0.3, 0.3, 1, 0.2, 0.1,
                             0.6, 0.2, 0.2, 1, 0.1,
                             0.6, 0.1, 0.1, 0.1, 1), nrow = 5, byrow = TRUE)
      Zero.Mat = matrix(0, nrow = 5, ncol = p-5)
      Sig = rbind(cbind(dependt.Cov, Zero.Mat), cbind(t(Zero.Mat), diag(p-5)))
      # Main effects only
      X.m1 = SimDesign::rmvnorm(n, mean = rep(0, p), sigma = Sig)
      colnames(X.m1) = paste0("X", 1:p)
      
      # Interaction effects
      id1 = unlist(sapply(1:p, function(x) x:p))
      id2 = rep(1:p, p:1)
      X.i = sapply(1:length(id1), function(x) X.m1[, id2[x]]*X.m1[, id1[x]])
      
      colnames(X.i) = sapply(1:length(id1), function(x) paste0("X", id2[x], "X", id1[x]))
      X = cbind(X.m1, X.i)
      
    }
    else{
      X.m2 = matrix(runif(n*p), nrow = n, ncol = p)
      colnames(X.m2) = paste0("X", 1:p)
      # Standardization (Mean Clustering)
      X.m2 = apply(X.m2, 2, scale, scale = FALSE) # The quadratic term of a centered variable can avoid collinearity
      
      # Interaction effects
      id1 = unlist(sapply(1:p, function(x) x:p))
      id2 = rep(1:p, p:1)
      X.i = sapply(1:length(id1), function(x) X.m2[, id2[x]]*X.m2[, id1[x]])
      colnames(X.i) = sapply(1:length(id1), function(x) paste0("X", id2[x], "X", id1[x]))
      
      X = cbind(X.m2, X.i)
      Xt = 2*(X - 0.5)
    }
  }
  
  # Models from Jain and Xu (2021)
  if (mod==5){
    prob1 = 1/(1 + exp(-(-0.5 + (1 + 0.2*X[,1] + 0.3*X[,2] + 0.4*X[,3] + 0.3*X[,1]*X[,2]))))
    Y = rbinom(n, 1, prob1) # Balanced Response case with main effect 
  }
  
  if (mod==6){
    prob1 = 1/(1 + exp(-(-2 + (1 + 0.2*X[,1] + 0.3*X[,2] + 0.4*X[,3] + 0.3*X[,1]*X[,2]))))
    Y = rbinom(n, 1, prob1) # Imbalanced Response case with main effect 
  }
  
  if (mod==7){
    prob2 = 1/(1 + exp(-(-0.5 + (1 + 0.4*X[,3] + 0.3*X[,1]*X[,2]))))
    Y = rbinom(n, 1, prob2) # Balanced Response case with no main effect (parent variable)
    
  }
  
  if (mod==8){
    prob2 = 1/(1 + exp(-(-2 + (1 + 0.4*X[,3] + 0.3*X[,1]*X[,2]))))
    Y = rbinom(n, 1, prob2) # Imbalanced Response case with no main effect (parent variable)
  }
  
  if (mod==9){
    prob6 = 1/(1 + exp(-(-0.5 + (1 + 0.4*X[,3] + 0.3*(X[,1]>0.5 & X[,2]>0.5)))))
    Y = rbinom(n, 1, prob6) # Balanced Response case )
  }
  
  if (mod==10){
    prob6 = 1/(1 + exp(-(-2 + (1 + 0.4*X[,3] + 0.3*(X[,1]>0.5 & X[,2]>0.5)))))
    Y = rbinom(n, 1, prob6) # Imbalanced Response case 
  }
  
  if (mod==11){
    prob6 = 1/(1 + exp(-(-0.5 + (1 + 0.2*X[,1] + 0.3*X[,2] + 0.3*X[,1]*X[,2] + 0.3*(X[,3]>0.5 & X[,4]>0.5)))))
    Y = rbinom(n, 1, prob6) # Balanced Response case 
  }
  
  if (mod==12){
    prob6 = 1/(1 + exp(-(-2 + (1 + 0.2*X[,1] + 0.3*X[,2] + 0.3*X[,1]*X[,2] + 0.3*(X[,3]>0.5 & X[,4]>0.5)))))
    Y = rbinom(n, 1, prob6) # Imbalanced Response case with
  }
  
  Dat = data.frame(cbind(Y, X))
  return(Dat)
}

####################################################################
############### Interaction Selection Algorithms ###################
####################################################################
LASSO.func <- function(train.DF, test.DF) {
  LASSO.time <- system.time(
    {
      LASSO.Mod <- cv.glmnet(x = as.matrix(train.DF[,-1]), y = as.matrix(train.DF$Y), alpha = 1, family = "binomial", type.measure = "class", standardize = FALSE)
      LASSO.coef = coef(LASSO.Mod, s = "lambda.min")
      LASSO.pred.tr = predict(LASSO.Mod, as.matrix(train.DF[,-1]), type = "class", s = "lambda.min")
      LASSO.pred.ts = predict(LASSO.Mod, as.matrix(test.DF[,-1]), type = "class", s = "lambda.min")
      LASSO.Conf.tr = confusionMatrix(factor(LASSO.pred.tr), factor(train.DF$Y), positive = "1", mode = "everything")
      LASSO.MCC.tr = round(mccr(train.DF$Y, factor(LASSO.pred.tr)), 4)
      LASSO.Conf.ts = confusionMatrix(factor(LASSO.pred.ts), factor(test.DF$Y), positive = "1", mode = "everything")
      LASSO.MCC.ts = round(mccr(test.DF$Y, factor(LASSO.pred.ts)), 4)
      rm(LASSO.Mod)
    }
  )
  return(list(LASSO.time, LASSO.coef, LASSO.Conf.tr, LASSO.MCC.tr, LASSO.Conf.ts, LASSO.MCC.ts))
}

NCP.func <- function(train.DF, test.DF, method) {
  
  NCP.time <- system.time(
    {
      NCP.Mod <- cv.ncvreg(X = as.matrix(train.DF[,-1]), y = as.matrix(train.DF$Y), family = "binomial", penalty = method)
      NCP.coef <- coef(NCP.Mod)
      NCP.pred.tr = predict(NCP.Mod, as.matrix(train.DF[,-1]), type = "class", s = "lambda.min")
      NCP.pred.ts = predict(NCP.Mod, as.matrix(test.DF[,-1]), type = "class", s = "lambda.min")
      NCP.Conf.tr = confusionMatrix(factor(NCP.pred.tr), factor(train.DF$Y), positive = "1", mode = "everything")
      NCP.MCC.tr = round(mccr(train.DF$Y, factor(NCP.pred.tr)), 4)
      NCP.Conf.ts = confusionMatrix(factor(NCP.pred.ts), factor(test.DF$Y), positive = "1", mode = "everything")
      NCP.MCC.ts = round(mccr(test.DF$Y, factor(NCP.pred.ts)), 4)
      rm(NCP.Mod)
    }
  )  
  return(list(NCP.time, NCP.coef, NCP.Conf.tr, NCP.MCC.tr, NCP.Conf.ts, NCP.MCC.ts))
}

###################################################

# RAMP ALGORITHM for three penalty functions and two types of interactions
RAMP.func <- function(train.DF, test.DF, penalty.val, hier.val){
  RAMP.Time <- system.time(
    temp.tr <- RAMP(X = as.matrix(train.DF[, -1]), y = as.matrix(train.DF[, 1]), family = "binomial",
                    penalty = penalty.val, gamma = NULL, inter = TRUE, hier = hier.val, eps = 1e-15,
                    tune = "EBIC", penalty.factor = rep(1, ncol(train.DF[, -1])), inter.penalty.factor = 1,
                    max.iter = 200, n.lambda = 100, ebic.gamma = 1, refit = TRUE, trace = FALSE))
  
  temp.fit <- predict(temp.tr, as.matrix(train.DF[, -1]), type = "class")
  temp.pred <- predict(temp.tr, as.matrix(test.DF[, -1]), type = "class")
  RAMP_Conf.tr <- confusionMatrix(factor(as.integer(temp.fit)), factor(train.DF[,1]), positive = "1", mode = "everything")
  RAMP.MCC.tr <- round(mccr(train.DF[,1], factor(as.integer(temp.fit))), 4)
  RAMP_Conf.ts <- confusionMatrix(factor(as.integer(temp.pred)), factor(test.DF[,1]), positive = "1", mode = "everything")
  RAMP.MCC.ts <- round(mccr(test.DF[,1], factor(as.integer(temp.pred))), 4)
  
  return(list(RAMP.Time, temp.tr$mainInd, temp.tr$interInd, temp.tr$beta.m, temp.tr$beta.i, RAMP_Conf.tr, RAMP.MCC.tr, RAMP_Conf.ts, RAMP.MCC.ts))
}

###################################################
RF.func <- function(train.DF, test.DF) {
  
  RF.Time <- system.time(
    { RF.imp = Boruta(Y~., data = train.DF, doTrace = 2, maxRuns = 300)
    RF.imp.var = rownames(attStats(RF.imp))[attStats(RF.imp)$decision=="Confirmed"]
    RF.tr <- randomForest(Y~., data= train.DF)
    
    RF.fit <- predict(RF.tr, as.matrix(train.DF[,-1]), "class")
    RF.pred <- predict(RF.tr, as.matrix(test.DF[,-1]), "class")
    RF.fit.tr <- ifelse(RF.fit >= 0.5, 1, 0)
    RF.pred.ts <- ifelse(RF.pred >= 0.5, 1, 0)
    RF.Conf.tr <- confusionMatrix(factor(RF.fit.tr), factor(train.DF$Y), positive = "1", mode = "everything")
    RF.MCC.tr <- round(mccr(test.DF[,1], factor(RF.fit.tr)), 4)
    RF.Conf.ts <- confusionMatrix(factor(RF.pred.ts), factor(test.DF$Y), positive = "1", mode = "everything")
    RF.MCC.ts <- round(mccr(test.DF[,1], factor(RF.pred.ts)), 4)
    rm(RF.tr)
    }
  )
  return(list(RF.Time, RF.imp.var, RF.Conf.tr, RF.MCC.tr, RF.Conf.ts, RF.MCC.ts))
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
                    rit.param = list(depth = 20, ntree = 100, nchild = 2, class.id = 1, class.cut = NULL),
                    varnames.grp = NULL,
                    n.bootstrap = 30,
                    bootstrap.forest = TRUE,
                    verbose = TRUE)
      
      iRF.fit <- predict(iRF.tr$rf.list[[5]], as.matrix(train.DF[,-1]), "class")
      iRF.pred <- predict(iRF.tr$rf.list[[5]], as.matrix(test.DF[,-1]), "class")
      iRF.fit.tr <- ifelse(iRF.fit >= 0.5, 1, 0)
      iRF.pred.ts <- ifelse(iRF.pred >= 0.5, 1, 0)
      iRF.inter <- iRF.tr$interaction[[5]]
      iRF.Conf.tr <- confusionMatrix(factor(iRF.fit.tr), factor(train.DF$Y), positive = "1", mode = "everything")
      iRF.MCC.tr <- round(mccr(test.DF[,1], factor(iRF.fit.tr)), 4)
      iRF.Conf.ts <- confusionMatrix(factor(iRF.pred.ts), factor(test.DF$Y), positive = "1", mode = "everything")
      iRF.MCC.ts <- round(mccr(test.DF[,1], factor(iRF.pred.ts)), 4)
      rm(iRF.tr)
    }
  )
  return(list(iRF.Time, iRF.inter, iRF.Conf.tr, iRF.MCC.tr, iRF.Conf.ts, iRF.MCC.ts))
}

##################################################################3
sim.func <- function(n, p, mod){
  
  # generating data and splitting to train/test
  DF.Sim <- DF.Gen(n = n.val, p = p.val, mod = mod)
  train.id = sample(n, floor(0.60*n), replace = FALSE, prob = NULL)
  
  train.DF = DF.Sim[train.id, ]
  test.DF = DF.Sim[-train.id, ]
  
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
  
  return(list(LASSO.res, SCAD.res, MCP.res, LASSO.W.res, LASSO.S.res, 
              SCAD.W.res, SCAD.S.res, MCP.W.res, MCP.S.res, RF.res, iRF.res))
}

###################################################################################

set.seed(12345)
iter = 100

n.val <- 500 #sample size
p.val <- c(25, 100, 500, 1000) #number of predictors

n <- 500
p <- c(25, 100, 500, 1000)
mod <- c(5, 7, 9, 11)      ## Balanced case

mod <- c(6, 8, 10, 12)     ## Imbalance case

system.time(
  {
    result2 = do.call(cbind, replicate(iter, sim.func(n = n.val, p = p.val, mod = mod)))
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

K <- 11*iter

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
  temp.name <- sapply(1:iter, function(x) c(names(dat[4, 1:K %% 11 == meth.pos][[x]]), dat[3, 1:K %% 11 == meth.pos][[x]]))
  temp <- sapply(1:iter, function(x) c(unlist(dat[4, 1:nc %% 11 == meth.pos][[x]]), unlist(dat[5, 1:nc %% 11 == meth.pos][[x]])))
  for (ii in 1:iter) {
    beta.est.RAMP[which(names(temp.beta) %in% temp.name[[ii]]), ii] = temp[[ii]]
    rownames(beta.est.RAMP) = names(temp.beta)
  }
  
  return(beta.est.RAMP)
}


Result.report.fun <- function(dat, mod = mod, p.val = p.val, n.meth = 11, cutoff.val = 0.7, iter = iter){
  
  {
    if (mod==5) beta = c(0.2, 0.3, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
    else if (mod==6) beta = c(0.2, 0.3, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
    else if (mod==7) beta = c(0, 0, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
    else if (mod==8) beta = c(0, 0, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))  # Need to update
    else if (mod==9) beta = c(0.2, 0.3, 0, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
    else if (mod==10) beta = c(0.2, 0.3, 0, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
    else if (mod==11) beta = c(0.2, 0.3, 0, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))  # Need to update
    else if (mod==12) beta = c(0.2, 0.3, 0, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
  }
  
  nc = ncol(dat)
  
  method <- c("LASSO", "SCAD", "MCP", "LASSO.W","LASSO.S", "SCAD.W", "SCAD.S", "MCP.W", "MCP.S", "RF", "iRF")
  metric <- c("Sens", "Spec", "FP", "FN", "L2")
  
  #### MSE
  
  Sim.time.result <- matrix(nrow = iter, ncol = n.meth)
  MSE.tr.result <- MSE.ts.result <- matrix(nrow=iter, ncol = 2*n.meth)
  Beta.result <- matrix(nrow = iter, ncol = 50)
  
  Sim.time.temp <- sapply(c(1:10,0), function(x) dat[1, ((1:nc) %% n.meth == x)])
  Sim.time.result <- matrix(sapply(Sim.time.temp, "[[", 3), nrow=iter)
  Sim.time <- rbind(apply(Sim.time.result, 2, mean), apply(Sim.time.result, 2, sd))
  colnames(Sim.time) <- method
  rownames(Sim.time) <- c("Mean", "SD")
  
  #### Beta evaluation
  
  LASSO_beta <- sapply(1:iter, function(x) beta.eval.func(beta, sapply(dat[2, 1:K %% 11 == 1], "[", -1)[,x]))
  SCAD_beta <- sapply(1:iter, function(x) beta.eval.func(beta, sapply(dat[2, 1:K %% 11 == 2], "[", -1)[,x]))
  MCP_beta <-  sapply(1:iter, function(x) beta.eval.func(beta, sapply(dat[2, 1:K %% 11 == 3], "[", -1)[,x]))
  
  # RAMP
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
    rbind( apply(MCP.S_beta,1, mean),apply(MCP.S_beta,1, sd))))
  colnames(Beta.result) <- metric
  Beta.result <- Beta.result %>%
    mutate(Method = rep(method[1:9], rep(2, length(method[1:9]))),
           Stat = rep(c("Mean", "SD"), 9)) %>%
    dplyr::select(Method, Stat, everything())
  
  ############################
  {
    if (mod == 5){
      true.pred <- c("X1", "X2", "X3", "X1X2")
      RF.beta.PPV = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:3])/length(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]]))
      RF.beta.Sens = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:3])/3)
      RF.beta <- data.frame(rbind(cbind( mean(RF.beta.Sens, na.rm = T), mean(RF.beta.PPV, na.rm = T)),
                                  cbind( sd(RF.beta.Sens, na.rm = T), sd(RF.beta.PPV, na.rm = T))))
      colnames(RF.beta) =c("Sens", "PPV")
      RF.beta <- RF.beta %>%
        mutate(Stat = c("Mean", "SD")) %>%
        dplyr::select(Stat, Sens, PPV)
    }
    else if (mod == 6) {
      true.pred <- c("X1", "X2", "X3", "X1X2")
      RF.beta.PPV = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:3])/length(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]]))
      RF.beta.Sens = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:3])/3)
      RF.beta <- data.frame(rbind(cbind( mean(RF.beta.Sens, na.rm = T), mean(RF.beta.PPV, na.rm = T)),
                                  cbind( sd(RF.beta.Sens, na.rm = T), sd(RF.beta.PPV, na.rm = T))))
      colnames(RF.beta) =c("Sens", "PPV")
      RF.beta <- RF.beta %>%
        mutate(Stat = c("Mean", "SD")) %>%
        dplyr::select(Stat, Sens, PPV)
    }
    else if (mod == 7) {
      true.pred <- c("X3", "X1X2")
      RF.beta.PPV = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1])/length(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]]))
      RF.beta.Sens = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1]))
      RF.beta <- data.frame(rbind(cbind( mean(RF.beta.Sens, na.rm = T), mean(RF.beta.PPV, na.rm = T)),
                                  cbind( sd(RF.beta.Sens, na.rm = T), sd(RF.beta.PPV, na.rm = T))))
      colnames(RF.beta) =c("Sens", "PPV")
      RF.beta <- RF.beta %>%
        mutate(Stat = c("Mean", "SD")) %>%
        dplyr::select(Stat, Sens, PPV)
    }
    else if (mod == 8) {
      true.pred <- c("X3", "X1X2")
      RF.beta.PPV = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1])/length(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]]))
      RF.beta.Sens = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1]))
      RF.beta <- data.frame(rbind(cbind( mean(RF.beta.Sens, na.rm = T), mean(RF.beta.PPV, na.rm = T)),
                                  cbind( sd(RF.beta.Sens, na.rm = T), sd(RF.beta.PPV, na.rm = T))))
      colnames(RF.beta) =c("Sens", "PPV")
      RF.beta <- RF.beta %>%
        mutate(Stat = c("Mean", "SD")) %>%
        dplyr::select(Stat, Sens, PPV)
    }
    else if (mod == 9) {
      true.pred <- c("X1", "X2", "X3", "X1X2")
      RF.beta.PPV = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:3])/length(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]]))
      RF.beta.Sens = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:3])/3)
      RF.beta <- data.frame(rbind(cbind( mean(RF.beta.Sens, na.rm = T), mean(RF.beta.PPV, na.rm = T)),
                                  cbind( sd(RF.beta.Sens, na.rm = T), sd(RF.beta.PPV, na.rm = T))))
      colnames(RF.beta) =c("Sens", "PPV")
      RF.beta <- RF.beta %>%
        mutate(Stat = c("Mean", "SD")) %>%
        dplyr::select(Stat, Sens, PPV)
    }
    else if (mod == 10) {
      true.pred <- c("X1", "X2", "X3", "X1X2")
      RF.beta.PPV = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:3])/length(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]]))
      RF.beta.Sens = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:3])/3)
      RF.beta <- data.frame(rbind(cbind( mean(RF.beta.Sens, na.rm = T), mean(RF.beta.PPV, na.rm = T)),
                                  cbind( sd(RF.beta.Sens, na.rm = T), sd(RF.beta.PPV, na.rm = T))))
      colnames(RF.beta) =c("Sens", "PPV")
      RF.beta <- RF.beta %>%
        mutate(Stat = c("Mean", "SD")) %>%
        dplyr::select(Stat, Sens, PPV)
    }
    else if (mod == 11) {
      true.pred <- c("X1", "X2", "X3", "X4", "X1X2")
      RF.beta.PPV = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:4])/length(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]]))
      RF.beta.Sens = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:4])/4)
      RF.beta <- data.frame(rbind(cbind( mean(RF.beta.Sens, na.rm = T), mean(RF.beta.PPV, na.rm = T)),
                                  cbind( sd(RF.beta.Sens, na.rm = T), sd(RF.beta.PPV, na.rm = T))))
      colnames(RF.beta) =c("Sens", "PPV")
      RF.beta <- RF.beta %>%
        mutate(Stat = c("Mean", "SD")) %>%
        dplyr::select(Stat, Sens, PPV)
    }
    else if (mod == 12) {
      true.pred <- c("X1", "X2", "X3", "X4", "X1X2")
      RF.beta.PPV = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:4])/length(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]]))
      RF.beta.Sens = sapply(1:iter, function(x) sum(sapply(dat[2, 1:nc %% 11 == 10], "[")[[x]] %in% true.pred[1:4])/4)
      RF.beta <- data.frame(rbind(cbind( mean(RF.beta.Sens, na.rm = T), mean(RF.beta.PPV, na.rm = T)),
                                  cbind( sd(RF.beta.Sens, na.rm = T), sd(RF.beta.PPV, na.rm = T))))
      colnames(RF.beta) =c("Sens", "PPV")
      RF.beta <- RF.beta %>%
        mutate(Stat = c("Mean", "SD")) %>%
        dplyr::select(Stat, Sens, PPV)
    }
  }
  
  iRF.inter = sapply(1:iter, function(x) unlist(sapply(dat[2, 1:nc %% 11 == 0], "[")[[x]])[unlist(sapply(dat[2, 1:nc %% 11 == 0], "[")[[x]])>cutoff.val])
  if (mod == 5){
    X1 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X2 <- sapply(1:iter, function(x) length(grep(true.pred[2], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[3], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X1, na.rm = T), mean(X2, na.rm = T), mean(X3, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X1, na.rm = T), sd(X2, na.rm = T), sd(X3, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD") 
  }
  else if (mod == 6){
    X1 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X2 <- sapply(1:iter, function(x) length(grep(true.pred[2], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[3], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X1, na.rm = T), mean(X2, na.rm = T), mean(X3, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X1, na.rm = T), sd(X2, na.rm = T), sd(X3, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD")
    
  }
  else if (mod == 7){
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X3, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X3, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD")
    
  }
  else if (mod == 8){
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X3, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X3, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD")
    
  }
  else if (mod == 9){
    X1 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X2 <- sapply(1:iter, function(x) length(grep(true.pred[2], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[3], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X1, na.rm = T), mean(X2, na.rm = T), mean(X3, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X1, na.rm = T), sd(X2, na.rm = T), sd(X3, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD") 
    
  }
  else if (mod == 10){
    X1 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X2 <- sapply(1:iter, function(x) length(grep(true.pred[2], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[3], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X1, na.rm = T), mean(X2, na.rm = T), mean(X3, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X1, na.rm = T), sd(X2, na.rm = T), sd(X3, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD") 
    
  }
  else if (mod == 11){
    X1 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X2 <- sapply(1:iter, function(x) length(grep(true.pred[2], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[3], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X4 <- sapply(1:iter, function(x) length(grep(true.pred[4], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    X3X4 <- sapply(1:iter, function(x) length(grep(true.pred[3], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X1, na.rm = T), mean(X2, na.rm = T), mean(X3, na.rm = T), mean(X4, na.rm = T), mean(X1X2, na.rm = T), mean(X3X4, na.rm = T)), c(sd(X1, na.rm = T), sd(X2, na.rm = T), sd(X3, na.rm = T), sd(X4, na.rm = T), sd(X1X2, na.rm = T), sd(X3X4, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD") 
    
  }
  else if (mod == 12){
    X1 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X2 <- sapply(1:iter, function(x) length(grep(true.pred[2], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[3], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X4 <- sapply(1:iter, function(x) length(grep(true.pred[4], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    X3X4 <- sapply(1:iter, function(x) length(grep(true.pred[3], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X1, na.rm = T), mean(X2, na.rm = T), mean(X3, na.rm = T), mean(X4, na.rm = T), mean(X1X2, na.rm = T), mean(X3X4, na.rm = T)), c(sd(X1, na.rm = T), sd(X2, na.rm = T), sd(X3, na.rm = T), sd(X4, na.rm = T), sd(X1X2, na.rm = T), sd(X3X4, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD") 
    
  }
  return(list(Sim.time, Beta.result, RF.beta, iRF.inter.result ))
}


Report <- Result.report.fun(dat = result2, mod = mod, p.val = p.val, n.meth = 11, cutoff.val = 0.7, iter = iter)

Report

##########################################################
#################### PREDICTION METRICS ##################
##########################################################
LASSO.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
LASSO.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
LASSO.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
LASSO.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
LASSO.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
LASSO.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
LASSO.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
LASSO.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
LASSO.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
LASSO.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
LASSO.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
LASSO.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
LASSO.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
LASSO.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
LASSO.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
LASSO.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

SCAD.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
SCAD.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
SCAD.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
SCAD.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
SCAD.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
SCAD.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
SCAD.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
SCAD.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
SCAD.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
SCAD.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
SCAD.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
SCAD.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
SCAD.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
SCAD.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
SCAD.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
SCAD.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

MCP.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
MCP.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
MCP.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
MCP.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
MCP.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
MCP.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
MCP.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
MCP.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
MCP.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
MCP.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
MCP.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
MCP.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
MCP.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
MCP.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
MCP.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
MCP.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

RLW.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
RLW.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
RLW.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
RLW.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RLW.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RLW.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
RLW.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
RLW.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
RLW.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
RLW.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
RLW.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
RLW.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RLW.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RLW.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
RLW.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
RLW.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

RLS.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
RLS.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
RLS.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
RLS.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RLS.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RLS.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
RLS.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
RLS.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
RLS.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
RLS.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
RLS.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
RLS.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RLS.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RLS.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
RLS.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
RLS.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

RSW.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
RSW.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
RSW.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
RSW.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RSW.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RSW.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
RSW.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
RSW.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
RSW.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
RSW.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
RSW.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
RSW.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RSW.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RSW.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
RSW.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
RSW.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

RSS.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
RSS.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
RSS.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
RSS.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RSS.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RSS.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
RSS.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
RSS.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
RSS.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
RSS.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
RSS.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
RSS.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RSS.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RSS.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
RSS.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
RSS.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

RMCPW.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPW.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

RMCPS.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
RMCPS.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

RF.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
RF.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
RF.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
RF.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RF.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
RF.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
RF.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
RF.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
RF.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
RF.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
RF.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
RF.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RF.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
RF.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
RF.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
RF.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

iRF.Acc.tr <- matrix(NA, nrow = iter, ncol = 1)
iRF.SENS.tr <- matrix(NA, nrow = iter, ncol = 1)
iRF.SPEC.tr <- matrix(NA, nrow = iter, ncol = 1)
iRF.PPV.tr <- matrix(NA, nrow = iter, ncol = 1)
iRF.NPV.tr <- matrix(NA, nrow = iter, ncol = 1)
iRF.F1.tr <- matrix(NA, nrow = iter, ncol = 1)
iRF.BalAcc.tr <- matrix(NA, nrow = iter, ncol = 1)
iRF.MCC.tr <- matrix(NA, nrow = iter, ncol = 1)
iRF.Acc.ts <- matrix(NA, nrow = iter, ncol = 1)
iRF.SENS.ts <- matrix(NA, nrow = iter, ncol = 1)
iRF.SPEC.ts <- matrix(NA, nrow = iter, ncol = 1)
iRF.PPV.ts <- matrix(NA, nrow = iter, ncol = 1)
iRF.NPV.ts <- matrix(NA, nrow = iter, ncol = 1)
iRF.F1.ts <- matrix(NA, nrow = iter, ncol = 1)
iRF.BalAcc.ts <- matrix(NA, nrow = iter, ncol = 1)
iRF.MCC.ts <- matrix(NA, nrow = iter, ncol = 1)

for (i in 1:iter){
  LASSO.Acc.tr[i,] = dat[[3, 1 + (i-1)*11]]$overall[1]
  LASSO.SENS.tr[i,] = dat[[3, 1 + (i-1)*11]]$byClass[1]
  LASSO.SPEC.tr[i,] = dat[[3, 1 + (i-1)*11]]$byClass[2]
  LASSO.PPV.tr[i,] = dat[[3, 1 + (i-1)*11]]$byClass[3]
  LASSO.NPV.tr[i,] = dat[[3, 1 + (i-1)*11]]$byClass[4]
  LASSO.F1.tr[i,] = dat[[3, 1 + (i-1)*11]]$byClass[7]
  LASSO.BalAcc.tr[i,] = dat[[3, 1 + (i-1)*11]]$byClass[11]
  LASSO.MCC.tr[i,] = dat[[4, 1 + (i-1)*11]]
  LASSO.Acc.ts[i,] = dat[[5, 1 + (i-1)*11]]$overall[1]
  LASSO.SENS.ts[i,] = dat[[5, 1 + (i-1)*11]]$byClass[1]
  LASSO.SPEC.ts[i,] = dat[[5, 1 + (i-1)*11]]$byClass[2]
  LASSO.PPV.ts[i,] = dat[[5, 1 + (i-1)*11]]$byClass[3]
  LASSO.NPV.ts[i,] = dat[[5, 1 + (i-1)*11]]$byClass[4]
  LASSO.F1.ts[i,] = dat[[5, 1 + (i-1)*11]]$byClass[7]
  LASSO.BalAcc.ts[i,] = dat[[5, 1 + (i-1)*11]]$byClass[11]
  LASSO.MCC.ts[i,] = dat[[6, 1 + (i-1)*11]]
  
  SCAD.Acc.tr[i,] = dat[[3, 2 + (i-1)*11]]$overall[1]
  SCAD.SENS.tr[i,] = dat[[3, 2 + (i-1)*11]]$byClass[1]
  SCAD.SPEC.tr[i,] = dat[[3, 2 + (i-1)*11]]$byClass[2]
  SCAD.PPV.tr[i,] = dat[[3, 2 + (i-1)*11]]$byClass[3]
  SCAD.NPV.tr[i,] = dat[[3, 2 + (i-1)*11]]$byClass[4]
  SCAD.F1.tr[i,] = dat[[3, 2 + (i-1)*11]]$byClass[7]
  SCAD.BalAcc.tr[i,] = dat[[3, 2 + (i-1)*11]]$byClass[11]
  SCAD.MCC.tr[i,] = dat[[4, 2 + (i-1)*11]]
  SCAD.Acc.ts[i,] = dat[[5, 2 + (i-1)*11]]$overall[1]
  SCAD.SENS.ts[i,] = dat[[5, 2 + (i-1)*11]]$byClass[1]
  SCAD.SPEC.ts[i,] = dat[[5, 2 + (i-1)*11]]$byClass[2]
  SCAD.PPV.ts[i,] = dat[[5, 2 + (i-1)*11]]$byClass[3]
  SCAD.NPV.ts[i,] = dat[[5, 2 + (i-1)*11]]$byClass[4]
  SCAD.F1.ts[i,] = dat[[5, 2 + (i-1)*11]]$byClass[7]
  SCAD.BalAcc.ts[i,] = dat[[5, 2 + (i-1)*11]]$byClass[11]
  SCAD.MCC.ts[i,] = dat[[6, 2 + (i-1)*11]]
  
  MCP.Acc.tr[i,] = dat[[3, 3 + (i-1)*11]]$overall[1]
  MCP.SENS.tr[i,] = dat[[3, 3 + (i-1)*11]]$byClass[1]
  MCP.SPEC.tr[i,] = dat[[3, 3 + (i-1)*11]]$byClass[2]
  MCP.PPV.tr[i,] = dat[[3, 3 + (i-1)*11]]$byClass[3]
  MCP.NPV.tr[i,] = dat[[3, 3 + (i-1)*11]]$byClass[4]
  MCP.F1.tr[i,] = dat[[3, 3 + (i-1)*11]]$byClass[7]
  MCP.BalAcc.tr[i,] = dat[[3, 3 + (i-1)*11]]$byClass[11]
  MCP.MCC.tr[i,] = dat[[4, 3 + (i-1)*11]]
  MCP.Acc.ts[i,] = dat[[5, 3 + (i-1)*11]]$overall[1]
  MCP.SENS.ts[i,] = dat[[5, 3 + (i-1)*11]]$byClass[1]
  MCP.SPEC.ts[i,] = dat[[5, 3 + (i-1)*11]]$byClass[2]
  MCP.PPV.ts[i,] = dat[[5, 3 + (i-1)*11]]$byClass[3]
  MCP.NPV.ts[i,] = dat[[5, 3 + (i-1)*11]]$byClass[4]
  MCP.F1.ts[i,] = dat[[5, 3 + (i-1)*11]]$byClass[7]
  MCP.BalAcc.ts[i,] = dat[[5, 3 + (i-1)*11]]$byClass[11]
  MCP.MCC.ts[i,] = dat[[6, 3 + (i-1)*11]]
  
  RLW.Acc.tr[i,] = dat[[6, 4 + (i-1)*11]]$overall[1]
  RLW.SENS.tr[i,] = dat[[6, 4 + (i-1)*11]]$byClass[1]
  RLW.SPEC.tr[i,] = dat[[6, 4 + (i-1)*11]]$byClass[2]
  RLW.PPV.tr[i,] = dat[[6, 4 + (i-1)*11]]$byClass[3]
  RLW.NPV.tr[i,] = dat[[6, 4 + (i-1)*11]]$byClass[4]
  RLW.F1.tr[i,] = dat[[6, 4 + (i-1)*11]]$byClass[7]
  RLW.BalAcc.tr[i,] = dat[[6, 4 + (i-1)*11]]$byClass[11]
  RLW.MCC.tr[i,] = dat[[7, 4 + (i-1)*11]]
  RLW.Acc.ts[i,] = dat[[8, 4 + (i-1)*11]]$overall[1]
  RLW.SENS.ts[i,] = dat[[8, 4 + (i-1)*11]]$byClass[1]
  RLW.SPEC.ts[i,] = dat[[8, 4 + (i-1)*11]]$byClass[2]
  RLW.PPV.ts[i,] = dat[[8, 4 + (i-1)*11]]$byClass[3]
  RLW.NPV.ts[i,] = dat[[8, 4 + (i-1)*11]]$byClass[4]
  RLW.F1.ts[i,] = dat[[8, 4 + (i-1)*11]]$byClass[7]
  RLW.BalAcc.ts[i,] = dat[[8, 4 + (i-1)*11]]$byClass[11]
  RLW.MCC.ts[i,] = dat[[9, 4 + (i-1)*11]]
  
  RLS.Acc.tr[i,] = dat[[6, 5 + (i-1)*11]]$overall[1]
  RLS.SENS.tr[i,] = dat[[6, 5 + (i-1)*11]]$byClass[1]
  RLS.SPEC.tr[i,] = dat[[6, 5 + (i-1)*11]]$byClass[2]
  RLS.PPV.tr[i,] = dat[[6, 5 + (i-1)*11]]$byClass[3]
  RLS.NPV.tr[i,] = dat[[6, 5 + (i-1)*11]]$byClass[4]
  RLS.F1.tr[i,] = dat[[6, 5 + (i-1)*11]]$byClass[7]
  RLS.BalAcc.tr[i,] = dat[[6, 5 + (i-1)*11]]$byClass[11]
  RLS.MCC.tr[i,] = dat[[7, 5 + (i-1)*11]]
  RLS.Acc.ts[i,] = dat[[8, 5 + (i-1)*11]]$overall[1]
  RLS.SENS.ts[i,] = dat[[8, 5 + (i-1)*11]]$byClass[1]
  RLS.SPEC.ts[i,] = dat[[8, 5 + (i-1)*11]]$byClass[2]
  RLS.PPV.ts[i,] = dat[[8, 5 + (i-1)*11]]$byClass[3]
  RLS.NPV.ts[i,] = dat[[8, 5 + (i-1)*11]]$byClass[4]
  RLS.F1.ts[i,] = dat[[8, 5 + (i-1)*11]]$byClass[7]
  RLS.BalAcc.ts[i,] = dat[[8, 5 + (i-1)*11]]$byClass[11]
  RLS.MCC.ts[i,] = dat[[9, 5 + (i-1)*11]]
  
  RSW.Acc.tr[i,] = dat[[6, 6 + (i-1)*11]]$overall[1]
  RSW.SENS.tr[i,] = dat[[6, 6 + (i-1)*11]]$byClass[1]
  RSW.SPEC.tr[i,] = dat[[6, 6 + (i-1)*11]]$byClass[2]
  RSW.PPV.tr[i,] = dat[[6, 6 + (i-1)*11]]$byClass[3]
  RSW.NPV.tr[i,] = dat[[6, 6 + (i-1)*11]]$byClass[4]
  RSW.F1.tr[i,] = dat[[6, 6 + (i-1)*11]]$byClass[7]
  RSW.BalAcc.tr[i,] = dat[[6, 6 + (i-1)*11]]$byClass[11]
  RSW.MCC.tr[i,] = dat[[7, 6 + (i-1)*11]]
  RSW.Acc.ts[i,] = dat[[8, 6 + (i-1)*11]]$overall[1]
  RSW.SENS.ts[i,] = dat[[8, 6 + (i-1)*11]]$byClass[1]
  RSW.SPEC.ts[i,] = dat[[8, 6 + (i-1)*11]]$byClass[2]
  RSW.PPV.ts[i,] = dat[[8, 6 + (i-1)*11]]$byClass[3]
  RSW.NPV.ts[i,] = dat[[8, 6 + (i-1)*11]]$byClass[4]
  RSW.F1.ts[i,] = dat[[8, 6 + (i-1)*11]]$byClass[7]
  RSW.BalAcc.ts[i,] = dat[[8, 6 + (i-1)*11]]$byClass[11]
  RSW.MCC.ts[i,] = dat[[9, 6 + (i-1)*11]]
  
  RSS.Acc.tr[i,] = dat[[6, 7 + (i-1)*11]]$overall[1]
  RSS.SENS.tr[i,] = dat[[6, 7 + (i-1)*11]]$byClass[1]
  RSS.SPEC.tr[i,] = dat[[6, 7 + (i-1)*11]]$byClass[2]
  RSS.PPV.tr[i,] = dat[[6, 7 + (i-1)*11]]$byClass[3]
  RSS.NPV.tr[i,] = dat[[6, 7 + (i-1)*11]]$byClass[4]
  RSS.F1.tr[i,] = dat[[6, 7 + (i-1)*11]]$byClass[7]
  RSS.BalAcc.tr[i,] = dat[[6, 7 + (i-1)*11]]$byClass[11]
  RSS.MCC.tr[i,] = dat[[7, 7 + (i-1)*11]]
  RSS.Acc.ts[i,] = dat[[8, 7 + (i-1)*11]]$overall[1]
  RSS.SENS.ts[i,] = dat[[8, 7 + (i-1)*11]]$byClass[1]
  RSS.SPEC.ts[i,] = dat[[8, 7 + (i-1)*11]]$byClass[2]
  RSS.PPV.ts[i,] = dat[[8, 7 + (i-1)*11]]$byClass[3]
  RSS.NPV.ts[i,] = dat[[8, 7 + (i-1)*11]]$byClass[4]
  RSS.F1.ts[i,] = dat[[8, 7 + (i-1)*11]]$byClass[7]
  RSS.BalAcc.ts[i,] = dat[[8, 7 + (i-1)*11]]$byClass[11]
  RSS.MCC.ts[i,] = dat[[9, 7 + (i-1)*11]]
  
  RMCPW.Acc.tr[i,] = dat[[6, 8 + (i-1)*11]]$overall[1]
  RMCPW.SENS.tr[i,] = dat[[6, 8 + (i-1)*11]]$byClass[1]
  RMCPW.SPEC.tr[i,] = dat[[6, 8 + (i-1)*11]]$byClass[2]
  RMCPW.PPV.tr[i,] = dat[[6, 8 + (i-1)*11]]$byClass[3]
  RMCPW.NPV.tr[i,] = dat[[6, 8 + (i-1)*11]]$byClass[4]
  RMCPW.F1.tr[i,] = dat[[6, 8 + (i-1)*11]]$byClass[7]
  RMCPW.BalAcc.tr[i,] = dat[[6, 8 + (i-1)*11]]$byClass[11]
  RMCPW.MCC.tr[i,] = dat[[7, 8 + (i-1)*11]]
  RMCPW.Acc.ts[i,] = dat[[8, 8 + (i-1)*11]]$overall[1]
  RMCPW.SENS.ts[i,] = dat[[8, 8 + (i-1)*11]]$byClass[1]
  RMCPW.SPEC.ts[i,] = dat[[8, 8 + (i-1)*11]]$byClass[2]
  RMCPW.PPV.ts[i,] = dat[[8, 8 + (i-1)*11]]$byClass[3]
  RMCPW.NPV.ts[i,] = dat[[8, 8 + (i-1)*11]]$byClass[4]
  RMCPW.F1.ts[i,] = dat[[8, 8 + (i-1)*11]]$byClass[7]
  RMCPW.BalAcc.ts[i,] = dat[[8, 8 + (i-1)*11]]$byClass[11]
  RMCPW.MCC.ts[i,] = dat[[9, 8 + (i-1)*11]]
  
  RMCPS.Acc.tr[i,] = dat[[6, 9 + (i-1)*11]]$overall[1]
  RMCPS.SENS.tr[i,] = dat[[6, 9 + (i-1)*11]]$byClass[1]
  RMCPS.SPEC.tr[i,] = dat[[6, 9 + (i-1)*11]]$byClass[2]
  RMCPS.PPV.tr[i,] = dat[[6, 9 + (i-1)*11]]$byClass[3]
  RMCPS.NPV.tr[i,] = dat[[6, 9 + (i-1)*11]]$byClass[4]
  RMCPS.F1.tr[i,] = dat[[6, 9 + (i-1)*11]]$byClass[7]
  RMCPS.BalAcc.tr[i,] = dat[[6, 9 + (i-1)*11]]$byClass[11]
  RMCPS.MCC.tr[i,] = dat[[7, 9 + (i-1)*11]]
  RMCPS.Acc.ts[i,] = dat[[8, 9 + (i-1)*11]]$overall[1]
  RMCPS.SENS.ts[i,] = dat[[8, 9 + (i-1)*11]]$byClass[1]
  RMCPS.SPEC.ts[i,] = dat[[8, 9 + (i-1)*11]]$byClass[2]
  RMCPS.PPV.ts[i,] = dat[[8, 9 + (i-1)*11]]$byClass[3]
  RMCPS.NPV.ts[i,] = dat[[8, 9 + (i-1)*11]]$byClass[4]
  RMCPS.F1.ts[i,] = dat[[8, 9 + (i-1)*11]]$byClass[7]
  RMCPS.BalAcc.ts[i,] = dat[[8, 9 + (i-1)*11]]$byClass[11]
  RMCPS.MCC.ts[i,] = dat[[9, 9 + (i-1)*11]]
  
  RF.Acc.tr[i,] = dat[[3, 10 + (i-1)*11]]$overall[1]
  RF.SENS.tr[i,] = dat[[3, 10 + (i-1)*11]]$byClass[1]
  RF.SPEC.tr[i,] = dat[[3, 10 + (i-1)*11]]$byClass[2]
  RF.PPV.tr[i,] = dat[[3, 10 + (i-1)*11]]$byClass[3]
  RF.NPV.tr[i,] = dat[[3, 10 + (i-1)*11]]$byClass[4]
  RF.F1.tr[i,] = dat[[3, 10 + (i-1)*11]]$byClass[7]
  RF.BalAcc.tr[i,] = dat[[3, 10 + (i-1)*11]]$byClass[11]
  RF.MCC.tr[i,] = dat[[4, 10 + (i-1)*11]]
  RF.Acc.ts[i,] = dat[[5, 10 + (i-1)*11]]$overall[1]
  RF.SENS.ts[i,] = dat[[5, 10 + (i-1)*11]]$byClass[1]
  RF.SPEC.ts[i,] = dat[[5, 10 + (i-1)*11]]$byClass[2]
  RF.PPV.ts[i,] = dat[[5, 10 + (i-1)*11]]$byClass[3]
  RF.NPV.ts[i,] = dat[[5, 10 + (i-1)*11]]$byClass[4]
  RF.F1.ts[i,] = dat[[5, 10 + (i-1)*11]]$byClass[7]
  RF.BalAcc.ts[i,] = dat[[5, 10 + (i-1)*11]]$byClass[11]
  RF.MCC.ts[i,] = dat[[6, 10 + (i-1)*11]]
  
  iRF.Acc.tr[i,] = dat[[3, 11 + (i-1)*11]]$overall[1]
  iRF.SENS.tr[i,] = dat[[3, 11 + (i-1)*11]]$byClass[1]
  iRF.SPEC.tr[i,] = dat[[3, 11 + (i-1)*11]]$byClass[2]
  iRF.PPV.tr[i,] = dat[[3, 11 + (i-1)*11]]$byClass[3]
  iRF.NPV.tr[i,] = dat[[3, 11 + (i-1)*11]]$byClass[4]
  iRF.F1.tr[i,] = dat[[3, 11 + (i-1)*11]]$byClass[7]
  iRF.BalAcc.tr[i,] = dat[[3, 11 + (i-1)*11]]$byClass[11]
  iRF.MCC.tr[i,] = dat[[4, 11 + (i-1)*11]]
  iRF.Acc.ts[i,] = dat[[5, 11 + (i-1)*11]]$overall[1]
  iRF.SENS.ts[i,] = dat[[5, 11 + (i-1)*11]]$byClass[1]
  iRF.SPEC.ts[i,] = dat[[5, 11 + (i-1)*11]]$byClass[2]
  iRF.PPV.ts[i,] = dat[[5, 11 + (i-1)*11]]$byClass[3]
  iRF.NPV.ts[i,] = dat[[5, 11 + (i-1)*11]]$byClass[4]
  iRF.F1.ts[i,] = dat[[5, 11 + (i-1)*11]]$byClass[7]
  iRF.BalAcc.ts[i,] = dat[[5, 11 + (i-1)*11]]$byClass[11]
  iRF.MCC.ts[i,] = dat[[6, 11 + (i-1)*11]]
  
  
  pred.metric = cbind(LASSO.Acc.tr, LASSO.SENS.tr, LASSO.SPEC.tr, LASSO.PPV.tr, LASSO.NPV.tr, LASSO.F1.tr, LASSO.BalAcc.tr, LASSO.MCC.tr,
                      LASSO.Acc.ts, LASSO.SENS.ts, LASSO.SPEC.ts, LASSO.PPV.ts, LASSO.NPV.ts, LASSO.F1.ts, LASSO.BalAcc.ts, LASSO.MCC.ts,
                      SCAD.Acc.tr, SCAD.SENS.tr, SCAD.SPEC.tr, SCAD.PPV.tr, SCAD.NPV.tr, SCAD.F1.tr, SCAD.BalAcc.tr, SCAD.MCC.tr,
                      SCAD.Acc.ts, SCAD.SENS.ts, SCAD.SPEC.ts, SCAD.PPV.ts, SCAD.NPV.ts, SCAD.F1.ts, SCAD.BalAcc.ts, SCAD.MCC.ts,
                      MCP.Acc.tr, MCP.SENS.tr, MCP.SPEC.tr, MCP.PPV.tr, MCP.NPV.tr, MCP.F1.tr, MCP.BalAcc.tr, MCP.MCC.tr,
                      MCP.Acc.ts, MCP.SENS.ts, MCP.SPEC.ts, MCP.PPV.ts, MCP.NPV.ts, MCP.F1.ts, MCP.BalAcc.ts, MCP.MCC.ts,
                      RLW.Acc.tr, RLW.SENS.tr, RLW.SPEC.tr, RLW.PPV.tr, RLW.NPV.tr, RLW.F1.tr, RLW.BalAcc.tr, RLW.MCC.tr,
                      RLW.Acc.ts, RLW.SENS.ts, RLW.SPEC.ts, RLW.PPV.ts, RLW.NPV.ts, RLW.F1.ts, RLW.BalAcc.ts, RLW.MCC.ts,
                      RLS.Acc.tr, RLS.SENS.tr, RLS.SPEC.tr, RLS.PPV.tr, RLS.NPV.tr, RLS.F1.tr, RLS.BalAcc.tr, RLS.MCC.tr,
                      RLS.Acc.ts, RLS.SENS.ts, RLS.SPEC.ts, RLS.PPV.ts, RLS.NPV.ts, RLS.F1.ts, RLS.BalAcc.ts, RLS.MCC.ts,
                      RSW.Acc.tr, RSW.SENS.tr, RSW.SPEC.tr, RSW.PPV.tr, RSW.NPV.tr, RSW.F1.tr, RSW.BalAcc.tr, RSW.MCC.tr,
                      RSW.Acc.ts, RSW.SENS.ts, RSW.SPEC.ts, RSW.PPV.ts, RSW.NPV.ts, RSW.F1.ts, RSW.BalAcc.ts, RSW.MCC.ts,
                      RSS.Acc.tr, RSS.SENS.tr, RSS.SPEC.tr, RSS.PPV.tr, RSS.NPV.tr, RSS.F1.tr, RSS.BalAcc.tr, RSS.MCC.tr,
                      RSS.Acc.ts, RSS.SENS.ts, RSS.SPEC.ts, RSS.PPV.ts, RSS.NPV.ts, RSS.F1.ts, RSS.BalAcc.ts, RSS.MCC.ts,
                      RMCPW.Acc.tr, RMCPW.SENS.tr, RMCPW.SPEC.tr, RMCPW.PPV.tr, RMCPW.NPV.tr, RMCPW.F1.tr, RMCPW.BalAcc.tr, RMCPW.MCC.tr,
                      RMCPW.Acc.ts, RMCPW.SENS.ts, RMCPW.SPEC.ts, RMCPW.PPV.ts, RMCPW.NPV.ts, RMCPW.F1.ts, RMCPW.BalAcc.ts, RMCPW.MCC.ts,
                      RMCPS.Acc.tr, RMCPS.SENS.tr, RMCPS.SPEC.tr, RMCPS.PPV.tr, RMCPS.NPV.tr, RMCPS.F1.tr, RMCPS.BalAcc.tr, RMCPS.MCC.tr,
                      RMCPS.Acc.ts, RMCPS.SENS.ts, RMCPS.SPEC.ts, RMCPS.PPV.ts, RMCPS.NPV.ts, RMCPS.F1.ts, RMCPS.BalAcc.ts, RMCPS.MCC.ts,
                      RF.Acc.tr, RF.SENS.tr, RF.SPEC.tr, RF.PPV.tr, RF.NPV.tr, RF.F1.tr, RF.BalAcc.tr, RF.MCC.tr,
                      RF.Acc.ts, RF.SENS.ts, RF.SPEC.ts, RF.PPV.ts, RF.NPV.ts, RF.F1.ts, RF.BalAcc.ts, RF.MCC.ts,
                      iRF.Acc.tr, iRF.SENS.tr, iRF.SPEC.tr, iRF.PPV.tr, iRF.NPV.tr, iRF.F1.tr, iRF.BalAcc.tr, iRF.MCC.tr,
                      iRF.Acc.ts, iRF.SENS.ts, iRF.SPEC.ts, iRF.PPV.ts, iRF.NPV.ts, iRF.F1.ts, iRF.BalAcc.ts, iRF.MCC.ts)
}

colnames(pred.metric) = c("LASSO.Acc.tr", "LASSO.SENS.tr", "LASSO.SPEC.tr", "LASSO.PPV.tr", "LASSO.NPV.tr", "LASSO.F1.tr", "LASSO.BalAcc.tr", "LASSO.MCC.tr",
                          "LASSO.Acc.ts", "LASSO.SENS.ts", "LASSO.SPEC.ts", "LASSO.PPV.ts", "LASSO.NPV.ts", "LASSO.F1.ts", "LASSO.BalAcc.ts", "LASSO.MCC.ts",
                          "SCAD.Acc.tr", "SCAD.SENS.tr", "SCAD.SPEC.tr", "SCAD.PPV.tr", "SCAD.NPV.tr", "SCAD.F1.tr", "SCAD.BalAcc.tr", "SCAD.MCC.tr",
                          "SCAD.Acc.ts", "SCAD.SENS.ts", "SCAD.SPEC.ts", "SCAD.PPV.ts", "SCAD.NPV.ts", "SCAD.F1.ts", "SCAD.BalAcc.ts", "SCAD.MCC.ts",
                          "MCP.Acc.tr", "MCP.SENS.tr", "MCP.SPEC.tr", "MCP.PPV.tr", "MCP.NPV.tr", "MCP.F1.tr", "MCP.BalAcc.tr", "MCP.MCC.tr",
                          "MCP.Acc.ts", "MCP.SENS.ts", "MCP.SPEC.ts", "MCP.PPV.ts", "MCP.NPV.ts", "MCP.F1.ts", "MCP.BalAcc.ts", "MCP.MCC.ts",
                          "RLW.Acc.tr", "RLW.SENS.tr", "RLW.SPEC.tr", "RLW.PPV.tr", "RLW.NPV.tr", "RLW.F1.tr", "RLW.BalAcc.tr", "RLW.MCC.tr",
                          "RLW.Acc.ts", "RLW.SENS.ts", "RLW.SPEC.ts", "RLW.PPV.ts", "RLW.NPV.ts", "RLW.F1.ts", "RLW.BalAcc.ts", "RLW.MCC.ts",
                          "RLS.Acc.tr", "RLS.SENS.tr", "RLS.SPEC.tr", "RLS.PPV.tr", "RLS.NPV.tr", "RLS.F1.tr", "RLS.BalAcc.tr", "RLS.MCC.tr",
                          "RLS.Acc.ts", "RLS.SENS.ts", "RLS.SPEC.ts", "RLS.PPV.ts", "RLS.NPV.ts", "RLS.F1.ts", "RLS.BalAcc.ts", "RLS.MCC.ts",
                          "RSW.Acc.tr", "RSW.SENS.tr", "RSW.SPEC.tr", "RSW.PPV.tr", "RSW.NPV.tr", "RSW.F1.tr", "RSW.BalAcc.tr", "RSW.MCC.tr",
                          "RSW.Acc.ts", "RSW.SENS.ts", "RSW.SPEC.ts", "RSW.PPV.ts", "RSW.NPV.ts", "RSW.F1.ts", "RSW.BalAcc.ts", "RSW.MCC.ts",
                          "RSS.Acc.tr", "RSS.SENS.tr", "RSS.SPEC.tr", "RSS.PPV.tr", "RSS.NPV.tr", "RSS.F1.tr", "RSS.BalAcc.tr", "RSS.MCC.tr",
                          "RSS.Acc.ts", "RSS.SENS.ts", "RSS.SPEC.ts", "RSS.PPV.ts", "RSS.NPV.ts", "RSS.F1.ts", "RSS.BalAcc.ts", "RSS.MCC.ts",
                          "RMCPW.Acc.tr", "RMCPW.SENS.tr", "RMCPW.SPEC.tr", "RMCPW.PPV.tr", "RMCPW.NPV.tr", "RMCPW.F1.tr", "RMCPW.BalAcc.tr", "RMCPW.MCC.tr",
                          "RMCPW.Acc.ts", "RMCPW.SENS.ts", "RMCPW.SPEC.ts", "RMCPW.PPV.ts", "RMCPW.NPV.ts", "RMCPW.F1.ts", "RMCPW.BalAcc.ts", "RMCPW.MCC.ts",
                          "RMCPS.Acc.tr", "RMCPS.SENS.tr", "RMCPS.SPEC.tr", "RMCPS.PPV.tr", "RMCPS.NPV.tr", "RMCPS.F1.tr", "RMCPS.BalAcc.tr", "RMCPS.MCC.tr",
                          "RMCPS.Acc.ts", "RMCPS.SENS.ts", "RMCPS.SPEC.ts", "RMCPS.PPV.ts", "RMCPS.NPV.ts", "RMCPS.F1.ts", "RMCPS.BalAcc.ts", "RMCPS.MCC.ts",
                          "RF.Acc.tr", "RF.SENS.tr", "RF.SPEC.tr", "RF.PPV.tr", "RF.NPV.tr", "RF.F1.tr", "RF.BalAcc.tr", "RF.MCC.tr",
                          "RF.Acc.ts", "RF.SENS.ts", "RF.SPEC.ts", "RF.PPV.ts", "RF.NPV.ts", "RF.F1.ts", "RF.BalAcc.ts", "RF.MCC.ts",
                          "iRF.Acc.tr", "iRF.SENS.tr", "iRF.SPEC.tr", "iRF.PPV.tr", "iRF.NPV.tr", "iRF.F1.tr", "iRF.BalAcc.tr", "iRF.MCC.tr",
                          "iRF.Acc.ts", "iRF.SENS.ts", "iRF.SPEC.ts", "iRF.PPV.ts", "iRF.NPV.ts", "iRF.F1.ts", "iRF.BalAcc.ts", "iRF.MCC.ts")

df.metrics <- pred.metric

PredRes <- as.data.frame(df.metrics)

PredMetric <- PredRes %>% 
  dplyr::summarise_each(funs(mean(.,na.rm = T), sd(.,na.rm = T)))

Metric <- PredMetric %>% 
  tidyr::gather(key = Pred, value = value) 
DF.sep <- separate(Metric,
                   col = Pred,
                   sep = "_",
                   into = c("Model", "Metrics"))

DF.Metrics <- DF.sep %>% 
  spread(key = Metrics, value = value)

# DF.Metrics




















