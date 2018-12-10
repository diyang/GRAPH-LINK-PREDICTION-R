# Clean workspace
rm(list=ls())
cat('\014')

require(mxnet)
#windows
#setwd("I:/Desktop/R/SAGE-GRAPH-R/graph_link_prediction")
#Mac
setwd("~/Documents/GRAPH-LINK-PREDICTION-R")
source("model.R")
source("utils.R")
source("train.R")

# load graph
org.graph.input <- loaddata.cora()
data <-as.matrix(org.graph.input$content[, -which(names(org.graph.input$content) %in% c("paper_id", "class"))])
org.graph.input[["features"]] <- data
adj <- org.graph.input$adjmatrix
edge.list <- as_edgelist(org.graph.input$graph)

# deliberately extract some edges, and re-construct graph
set.seed(123)
batch.size <- 50
num.edge <- dim(edge.list)[1]
pos.pair.indices.pool <- sample(c(1:num.edge), (batch.size*3), replace=FALSE)
positive.nodes.pairs <- list()
count <- 1
for(i in pos.pair.indices.pool){
  node.pair <- list(a=edge.list[i,1], b=edge.list[i,2])
  positive.nodes.pairs[[count]] <- node.pair
  count <- count + 1
}

num.node <- dim(adj)[1]
neg.pair.indices.pool <- sample(c(1:num.node), (batch.size*3), replace=FALSE)
negative.nodes.pairs <- list()
count <- 1
negative.edge.list <- c()
for(i in neg.pair.indices.pool){
  null.edge <- which(adj[i,] == 0)
  num.null.edge <- length(null.edge)
  if(num.null.edge > 0){
    neg.index <- sample(c(1:num.null.edge), 1)
    node.pair <- list(a=i, b=null.edge[neg.index])
    negative.nodes.pairs[[count]] <- node.pair
    count <- count + 1
  }
}

positive.label <- rep(1, length(positive.nodes.pairs))
negative.label <- rep(2, length(negative.nodes.pairs))

nodes.pairs <- c(positive.nodes.pairs, negative.nodes.pairs)
pairs.label <- c(positive.label, negative.label)

num.pairs <- length(nodes.pairs)
shuffled.indices <- sample(c(1:num.pairs))
shuffled.nodes.pairs <- nodes.pairs[shuffled.indices]
shuffled.pairs.label <- pairs.label[shuffled.indices]

new.edge.list <- edge.list[-(pos.pair.indices.pool),]
new.graph <- graph_from_edgelist(new.edge.list, directed = FALSE)

adjmatrix <- as_adj(new.graph, type = 'both', edges=FALSE, sparse = igraph_opt("sparsematrices"))
adjmatrix[which(adjmatrix>1)] <- 1
D.sqrt <- sqrt(colSums(adjmatrix))
A.tilde <- adjmatrix + Diagonal(dim(adjmatrix)[1])
P <- diag(D.sqrt)%*%A.tilde%*% diag(D.sqrt)

new.graph.input <- list(adjmatrix = adjmatrix, P = P, Atilde = A.tilde, Dsqrt = D.sqrt, graph = new.graph)
new.graph.input[['features']] <- data

K <- 2
num.hidden <- c(20,20,20)
max.nodes <- 50
input.size <- dim(data)[2]

gcn.sym <- GCN.layer.link.prediction(input.size,
                                     K,
                                     max.nodes, 
                                     batch.size, 
                                     num.hidden)

gcn.model <- GCN.link.setup.model(gcn.sym, 
                                  max.nodes,
                                  input.size,
                                  batch.size,
                                  K=K)

train.nodes.pairs <- shuffled.nodes.pairs#[1:(4*batch.size)]
train.pairs.label <- (shuffled.pairs.label-1)#[1:(4*batch.size)]

valid.nodes.pairs <- shuffled.nodes.pairs[(4*batch.size+1):num.pairs]
valid.pairs.label <- shuffled.pairs.label[(4*batch.size+1):num.pairs]

train.data <- list(nodes.pairs=train.nodes.pairs, pairs.label=train.pairs.label)
valid.data <- list(nodes.pairs=valid.nodes.pairs, pairs.label=valid.pairs.label)

learning.rate <- 0.01
weight.decay <- 0
clip.gradient <- 1
momentum <- 0.9
optimizer <- 'sgd'
lr.scheduler <- mx.lr_scheduler.FactorScheduler(step = 480, factor=0.5, stop_factor_lr = 1e-3)

gcn.model.trained <- GCN.link.trian.model(model = gcn.model,
                                          graph.input = new.graph.input,
                                          train.data = train.data,
                                          valid.data = NULL,
                                          num.epoch = 100,
                                          learning.rate = learning.rate,
                                          momentum= momentum,
                                          weight.decay = weight.decay,
                                          clip.gradient = clip.gradient,
                                          optimizer = optimizer)
                                          #lr.scheduler = lr.scheduler)