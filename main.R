require(mxnet)
#windows
#setwd("I:/Desktop/R/SAGE-GRAPH-R/graph_link_prediction")
#Mac
setwd("~/Documents/GRAPH-LINK-PREDICTION-R")
source("model.R")
source("utils.R")
source("train.R")

# model establish


graph.input <- loaddata.cora()
data <-as.matrix(graph.input$content[, -which(names(graph.input$content) %in% c("paper_id", "class"))])
graph.input[["features"]] <- data
adj <- graph.input$adjmatrix
edge.list <- as_edgelist(graph.input$graph)
num.edge <- dim(edge.list)[1]
pair.indices.pool <- sample(c(1:num.edge), batch.size, replace=FALSE)
nodes.pairs <- list()
count <- 1
for(i in pair.indices.pool){
  node.pair <- list(a=edge.list[i,1], b=edge.list[i,2])
  nodes.pairs[[count]] <- node.pair
  count <- count + 1
}

K <- 2
batch.size <- 100
num.hidden <- c(20,20)
input.size <- 30
max.nodes <- 300
num.filters <- c(10,5)
input.size <- dim(data)[2]

gcn.pair.train.input <- Graph.enclose.encode(nodes.pairs, adj, K, max.nodes)






level.label <- unique(graph.input$content$class)
num.label <- length(unique(level.label))
label.text <-graph.input$content$class 
label <- rep(0, length(label.text))
for(i in 1:num.label){
  class.ind <- which(label.text == level.label[i])
  label[class.ind] <- i
}
data <-as.matrix(graph.input$content[, -which(names(graph.input$content) %in% c("paper_id", "class"))])
graph.input[["features"]] <- list(data=data, label=label) 




K <- 2
input.shape <- list()
input.shape[['label']] <- c(batch.size)
input.shape[['data']] <- c(batch.size, input.size, max.nodes)
for(i in 1:K){
  variable.P <- paste0("P.",i,".tilde")
  input.shape[[variable.P]] <- c(max.nodes, max.nodes)
}

gcn.sym <- GCN.layer.link.prediction(input.size, 
                                     max.nodes, 
                                     batch.size, 
                                     num.hidden, 
                                     num.filters)

gcn.model <- GCN.link.setup.model(gcn.sym, 
                                  max.nodes,
                                  input.size,
                                  batch.size,
                                  K=2)

# training process
num.nodes.train <- floor((length(label) * 0.3)/batch.size)*batch.size
num.nodes.valid <- floor((length(label) * 0.1)/batch.size)*batch.size
nodes.pool <- sample(c(1:length(label)), (num.nodes.train+num.nodes.valid), replace=FALSE)
nodes.train.pool <- sort(nodes.pool[1:num.nodes.train])
nodes.valid.pool <- sort(nodes.pool[(num.nodes.train+1):(num.nodes.train+num.nodes.valid)])

learning.rate <- 0.005
weight.decay <- 0
clip.gradient <- 1
optimizer <- 'sgd'
lr.scheduler <- mx.lr_scheduler.FactorScheduler(step = 480, factor=0.5, stop_factor_lr = 1e-3)
gcn.model.trained <- GCN.trian.model(model = gcn.model,
                                     graph.input = graph.input,
                                     nodes.train.pool = nodes.train.pool,
                                     nodes.valid.pool = nodes.valid.pool,
                                     num.epoch = 100,
                                     learning.rate = learning.rate,
                                     weight.decay = weight.decay,
                                     clip.gradient = clip.gradient,
                                     optimizer = optimizer)
                                     #lr.scheduler = lr.scheduler)