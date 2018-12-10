require(mxnet)
setwd("~/Documents/GRAPH-LINK-PREDICTION-R")
source("model.R")
source("utils.R")
source("train.R")
max.nodes <- 3
input.size <- 20
batch.size <- 4
K <- 2

model1.sym <- m1(input.size,
             max.nodes,
             batch.size,
             K) 

model1.mod <- m1.setup(model1.sym,
                       max.nodes,
                       input.size,
                       batch.size,
                       K)


data <- array(dim=c(4,20,3))
data[1,,] <- matrix(1,20,3)
data[2,,] <- matrix(2,20,3)
data[3,,] <- matrix(3,20,3)
data[4,,] <- matrix(4,20,3)

tp.1 <- array(dim=c(4,3,3))
tp.1[1,,] <- diag(3)
tp.1[2,,] <- diag(3)
tp.1[3,,] <- diag(3)
tp.1[4,,] <- diag(3)

tp.2 <- array(dim=c(4,3,3))
tp.2[1,,] <- 2*diag(3)
tp.2[2,,] <- 2*diag(3)
tp.2[3,,] <- 2*diag(3)
tp.2[4,,] <- 2*diag(3)

tp.3 <- array(dim=c(4,3,3))
tp.3[1,,] <- 3*diag(3)
tp.3[2,,] <- 3*diag(3)
tp.3[3,,] <- 3*diag(3)
tp.3[4,,] <- 3*diag(3)

input.data <- list()
input.data[['data']] <- mx.nd.array(data)
input.data[['P.1.tilde']] <- mx.nd.array(tp.1)
input.data[['P.2.tilde']] <- mx.nd.array(tp.2)
input.data[['P.3.tilde']] <- mx.nd.array(tp.3)

mx.exec.update.arg.arrays(model1.mod$gcn.exec, input.data, match.name = TRUE)
mx.exec.forward(model1.mod$gcn.exec, is.train = TRUE)

output <- as.array(model1.mod$gcn.exec$ref.outputs[[1]])
