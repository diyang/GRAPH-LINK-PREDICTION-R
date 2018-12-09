require(mxnet)
source("model.R")
source("utils.R")
source("train.R")
a <- array(dim=c(4,2,3))
a[1,,] <- matrix(1,2,3)
a[2,,] <- matrix(2,2,3)
a[3,,] <- matrix(3,2,3)
a[4,,] <- matrix(4,2,3)

aa<-mx.nd.array(a)
bb <- mx.nd.Reshape(mx.nd.transpose(data=aa, axes=c(1,0,2)), shape=c(2, 12))
b<-as.array(bb)
cc <- mx.nd.Reshape(bb, shape=c(2,3,4))

c <- as.array(cc)

dd <- mx.nd.transpose(cc, axes=c(0,2,1))

d <- as.array(dd)


ee <- mx.nd.Concat(data=c(aa,dd), num_args = 2, dim=1)

e <- as.array(ee)

ff <- mx.nd.transpose(ee, axes = c(0,1,2))

f <- as.array(ff)

gg <- mx.nd.Reshape(ff, shape=c(3,4,1,4))

g <- as.array(gg)


max.nodes <- 3
input.size <- 2
batch.size <- 4
K <- 2

input.shape <- list()
input.shape[['data']] <- c(batch.size, input.size, max.nodes)
for(i in 1:K){
  variable <- paste0("P.",i,".tilde")
  input.shape[[variable]] <- c(batch.size, max.nodes, max.nodes)
}

model1.sym <- m1(input.size,
             max.nodes,
             batch.size,
             K) 

model1.mod <- m1.setup(model1.sym,
                       max.nodes,
                       input.size,
                       batch.size,
                       K)


data <- array(dim=c(4,2,3))
data[1,,] <- matrix(1,2,3)
data[2,,] <- matrix(2,2,3)
data[3,,] <- matrix(3,2,3)
data[4,,] <- matrix(4,2,3)

tp.1 <- array(dim=c(4,3,3))
tp.1[1,,] <- 2*diag(3)
tp.1[2,,] <- 2*diag(3)
tp.1[3,,] <- 2*diag(3)
tp.1[4,,] <- 2*diag(3)

tp.2 <- array(dim=c(4,3,3))
tp.2[1,,] <- 3*diag(3)
tp.2[2,,] <- 3*diag(3)
tp.2[3,,] <- 3*diag(3)
tp.2[4,,] <- 3*diag(3)

input.data <- list()
input.data[['data']] <- mx.nd.array(data)
input.data[['P.1.tilde']] <- mx.nd.array(tp.1)
input.data[['P.2.tilde']] <- mx.nd.array(tp.2)

mx.exec.update.arg.arrays(model1.mod$gcn.exec, input.data, match.name = TRUE)
mx.exec.forward(model1.mod$gcn.exec, is.train = TRUE)

model1.mod$gcn.exec$ref.outputs
