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

(core.num <- detectCores())

DF.Gen <- function(n, p, mod){
  # Models from Duroux and Scornet (2018)
  {
    if(mod < 5){
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
  if (mod==1){
    Y = 1 + 0.2*X[,1] + 0.3*X[,2] + 0.4*X[,3] + 0.3*X[,1]*X[,2] + rnorm(n, mean = 0, sd = sqrt(.25))
  }
  
  if (mod==2){
    Y = 1 + 0.4*X[,3] + 0.3*X[,1]*X[,2] + rnorm(n,mean = 0, sd = sqrt(0.25))
  }
  
  if (mod==3){
    Y = 1 + 0.4*X[,3] + 0.3*(X[,1]>0.5 & X[,2]>0.5) + rnorm(n, mean = 0, sd = sqrt(.25))
  }
  
  if (mod==4){
    Y = 1 + 0.2*X[,1] + 0.3*X[,2] + 0.3*X[,1]*X[,2] + 0.3*(X[,3]>0.5 & X[,4]>0.5) + rnorm(n, mean = 0, sd = sqrt(.25))
  }
  
  Dat = data.frame(cbind(Y, X))
  return(Dat)
}

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
                    max.iter = 200, n.lambda = 100, ebic.gamma = 1, refit = TRUE, trace = FALSE))
 
  temp.fit <- predict(temp.tr, as.matrix(train.DF[, -1]))
  temp.pred <- predict(temp.tr, as.matrix(test.DF[, -1]))
  mse.tr <- mse.fun(temp.fit, train.DF[,1])
  mse.ts <- mse.fun(temp.pred, test.DF[,1])
  
  return(list(RAMP.Time, temp.tr$mainInd, temp.tr$interInd, temp.tr$beta.m, temp.tr$beta.i, mse.tr, mse.ts))
}

###################################################
RF.func <- function(train.DF, test.DF) {
  
  RF.Time <- system.time(
    { RF.imp = Boruta(Y~., data = train.DF, doTrace = 2, maxRuns = 300)
    RF.imp.var = rownames(attStats(RF.imp))[attStats(RF.imp)$decision=="Confirmed"]
    RF.tr <- randomForest(Y~., data= train.DF)
    
    RF.fit <- predict(RF.tr, as.matrix(train.DF[,-1]))
    RF.pred <- predict(RF.tr, as.matrix(test.DF[,-1]))
    RF.mse.tr <- mse.fun(RF.fit, train.DF$Y)
    RF.mse.ts <- mse.fun(RF.pred, test.DF$Y)
    rm(RF.tr)
    }
  )
  return(list(RF.Time, RF.imp.var, RF.mse.tr, RF.mse.ts))
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

##################################################################
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
  
  # Hier.res = Hier.func(train.DF1, test.DF1)
  
  RF.res = RF.func(train.DF, test.DF)
  iRF.res = iRF.func(train.DF, test.DF)                                        # Main effects only
  
  return(list(LASSO.res, SCAD.res, MCP.res, LASSO.W.res, LASSO.S.res, SCAD.W.res,
              SCAD.S.res, MCP.W.res, MCP.S.res, RF.res, iRF.res))
}

###################################################################################

set.seed(12345)
iter = 100

n.val <- 500       # sample size
p.val <- 25        # c(25, 100, 500, 1000) #number of predictors

p <- 25            # c(25, 100, 500, 1000)
n <- 500

mod <- 1           # c(1,2,3,4)
n.meth = 11


system.time(
  {
    result2 = do.call(cbind, replicate(iter, sim.func(n=n.val, p=p.val, mod = mod)))
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

K <- n.meth*iter

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
  temp.name <- sapply(1:iter, function(x) c(names(dat[4, 1:(11*iter) %% 11 == meth.pos][[x]]), dat[3, 1:(11*iter) %% 11 == meth.pos][[x]]))
  temp <- sapply(1:iter, function(x) c(unlist(dat[4,1:nc %% 11 == meth.pos][[x]]), unlist(dat[5,1:nc %% 11 == meth.pos][[x]])))
  for (ii in 1:iter) {
    beta.est.RAMP[which(names(temp.beta) %in% temp.name[[ii]]), ii] = temp[[ii]]
    rownames(beta.est.RAMP) = names(temp.beta)
  }
  
  return(beta.est.RAMP)
}

Result.report.fun <- function(dat, mod = mod, p.val = p.val, n.meth = 11, cutoff.val = 0.7, iter = iter){
  
  {
    if (mod==1) beta =  c(0.2, 0.3, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
    else if (mod==2) beta = c(0, 0, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
    else if (mod==3) beta = c(0, 0, 0.4, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2)) 
    else if (mod==4) beta = c(0.2, 0.3, 0, rep(0,p.val-3), 0, 0.3, rep(0, p.val*(p.val+1)/2-2))
  }
  
  nc = ncol(dat)
  
  method <- c("LASSO", "SCAD", "MCP", "LASSO.W","LASSO.S", "SCAD.W", "SCAD.S", "MCP.W", "MCP.S", "RF", "iRF")
  metric <- c("Sens", "Spec", "FP", "FN", "L2")
  
  #### MSE
  
  Sim.time.result <- matrix(nrow = iter, ncol = n.meth)
  MSE.tr.result <- MSE.ts.result <- matrix(nrow = iter, ncol = 2*n.meth)
  Beta.result <- matrix(nrow = iter, ncol = 50)
  
  Sim.time.temp <- sapply(c(1:10,0), function(x) dat[1, ((1:nc) %% n.meth == x)])
  Sim.time.result <- matrix(sapply(Sim.time.temp, "[[", 3), nrow = iter)
  Sim.time <- rbind(apply(Sim.time.result, 2, mean), apply(Sim.time.result, 2, sd))
  colnames(Sim.time) <- method
  rownames(Sim.time) <- c("Mean", "SD")
  
  MSE.tr.result <- data.frame(cbind( sapply(1:3, function(x) dat[3, ((1:nc) %% n.meth == x)]),
                                     sapply(4:9, function(x) dat[6, ((1:nc) %% n.meth == x)]),
                                     # sapply(10, function(x) dat[4, ((1:nc) %% n.meth == x)]),
                                     sapply(c(10,0), function(x) dat[3, ((1:nc) %% n.meth == x)])))
  
  MSE.ts.result <- data.frame(cbind( sapply(1:3, function(x) dat[4, ((1:nc) %% n.meth == x)]),
                                     sapply(4:9, function(x) dat[7, ((1:nc) %% n.meth == x)]),
                                     # sapply(10, function(x) dat[5, ((1:nc) %% n.meth == x)]),
                                     sapply(c(10,0), function(x) dat[4, ((1:nc) %% n.meth == x)])))
  
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
  
  LASSO_beta <- sapply(1:iter, function(x) beta.eval.func(beta, sapply(dat[2, 1:(11*iter) %% 11 == 1], "[", -1)[,x]))
  SCAD_beta <- sapply(1:iter, function(x) beta.eval.func(beta, sapply(dat[2, 1:(11*iter) %% 11 == 2], "[", -1)[,x]))
  MCP_beta <-  sapply(1:iter, function(x) beta.eval.func(beta, sapply(dat[2, 1:(11*iter) %% 11 == 3], "[", -1)[,x]))
  
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
    if (mod == 1){
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
    else if (mod == 2) {
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
    
    else if (mod == 3) {
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
    else if (mod == 4) {
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
  if (mod == 1){
    X1 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X2 <- sapply(1:iter, function(x) length(grep(true.pred[2], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[3], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X1, na.rm = T), mean(X2, na.rm = T), mean(X3, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X1, na.rm = T), sd(X2, na.rm = T), sd(X3, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD") 
  }
  else if (mod == 2){
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X3, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X3, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD")
    
  }
  else if (mod == 3){
    X1 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X2 <- sapply(1:iter, function(x) length(grep(true.pred[2], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[3], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X1, na.rm = T), mean(X2, na.rm = T), mean(X3, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X1, na.rm = T), sd(X2, na.rm = T), sd(X3, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD") 
    
  }
  else if (mod == 4){
    X1 <- sapply(1:iter, function(x) length(grep(true.pred[1], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X2 <- sapply(1:iter, function(x) length(grep(true.pred[2], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X3 <- sapply(1:iter, function(x) length(grep(true.pred[3], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X4 <- sapply(1:iter, function(x) length(grep(true.pred[4], names(iRF.inter[[x]])))/length(names(iRF.inter[[x]])))
    X1X2 <- sapply(1:iter, function(x) length(grep(true.pred[1], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    X3X4 <- sapply(1:iter, function(x) length(grep(true.pred[3], gsub("_","", names(iRF.inter[[x]]))))/length(names(iRF.inter[[x]])))
    iRF.inter.result <- data.frame(rbind( c(mean(X1, na.rm = T), mean(X2, na.rm = T), mean(X3, na.rm = T), mean(X4, na.rm = T), mean(X1X2, na.rm = T)), c(sd(X1, na.rm = T), sd(X2, na.rm = T), sd(X3, na.rm = T), sd(X4, na.rm = T), sd(X1X2, na.rm = T))))
    colnames(iRF.inter.result) = true.pred 
    rownames(iRF.inter.result) = c("Mean", "SD") 
    
  }
  return(list(Sim.time, MSE.result, Beta.result, RF.beta, iRF.inter.result))
}


Report <- Result.report.fun(dat = result2, mod = mod, p.val = p.val, n.meth = 11, cutoff.val = 0.7, iter = iter)

# Report

























