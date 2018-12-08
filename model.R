# Author: Di YANG
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

require('mxnet')

Graph.Convolution <- function(data,
                              neighbors=NULL,
                              tP, 
                              num.hidden,
                              dropout = 0)
{
  if(dropout > 0){
    data <- mx.symbol.Dropout(data=data, p=dropout)
  }
  if(is.null(neighbors)){
    neighbors <- data
  }
  data.aggerator <- mx.symbol.dot(tP, neighbors)
  conv.input <- mx.symbol.Concat(data = c(data, data.aggerator), num.args = 2, dim = 1)
  graph.output <- mx.symbol.FullyConnected(data=conv.input, num_hidden = num.hidden)
  graph.activation <- mx.symbol.Activation(data=graph.output, act.type='relu')
  graph.L2norm <- mx.symbol.L2Normalization(graph.activation)
  return(graph.L2norm)
}

Graph.Convolution.batch <- function(data, 
                                    tP,
                                    num.hidden,
                                    batch.size,
                                    max.nodes,
                                    dropout = 0)
{
  if(dropout > 0){
    data <- mx.symbol.Dropout(data=data, p=dropout)
  }
  data.slice <- mx.symbol.SliceChannel(data=data, num_outputs=batch.size, axis= 2, squeeze_axis=1)
  layer.tP.slice <- mx.symbol.SliceChannel(data=tP, num_outputs=batch.size, axis= 2, squeeze_axis=1)
  data.aggerator <- list()
  for(i in 1:batch.size){
    data.aggerator[[i]] <- mx.symbol.dot(layer.tP.slice[[i]], data.slice[[i]])
  }
  data.aggerator.concat <- mx.symbol.Concat(data=data.aggerator, num.args = batch.size, dim=0)
  data.slice.concat <- mx.symbol.Reshape(mx.symbol.transpose(data=data, axes=c(1,0,2)), shape=c(input.size, (batch.size*max.nodes)))
  conv.input <- mx.symbol.Concat(data = c(data.slice.concat, data.aggerator.concat), num.args = 2, dim = 1)
  graph.output <- mx.symbol.FullyConnected(data=conv.input, num_hidden = num.hidden)
  graph.activation <- mx.symbol.Activation(data=graph.output, act.type='relu')
  graph.L2norm.flatten <- mx.symbol.L2Normalization(graph.activation)
  graph.L2norm <- mx.symbol.transpose(mx.symbol.Reshape(data=graph.L2norm.flatten, shape=c(num.hidden, max.nodes, batch.size)), axes=c(0,2,1))
  return(graph.L2norm)
}

GCN.layer.node.classifiction <- function(num.hidden, num.label, dropout = 0){
  label <- mx.symbol.Variable('label')
  layer.tP <- list()
  layer.H  <- list()
  K <- length(num.hidden)
  for(i in 1:K){
    layer.tP[[i]] <- mx.symbol.Variable(paste0("P.",i,".tilde"))
    layer.H[[i]]  <- mx.symbol.Variable(paste0("H.",i,".tilde"))
  }
  layer.H[[K+1]]  <- mx.symbol.Variable(paste0("H.",(K+1),".tilde"))
  
  layer.outputs <- list()
  for(i in K:1){
    gcn.input <- layer.H[[i]]
    if(i == K){
      neighbour.input <- layer.H[[i+1]]
    }else{
      neighbour.input <- layer.outputs[[i+1]]
    }
    layer.outputs[[i]] <- Graph.Convolution(data=gcn.input,
                                            neighbors = neighbour.input,
                                            tP=layer.tP[[i]], 
                                            num.hidden = num.hidden[i])
  }
  
  fc <- mx.symbol.FullyConnected(data=layer.outputs[[1]], num.hidden=num.label)
  loss.all <- mx.symbol.SoftmaxOutput(data=fc, label=label, name="sm")
  return(loss.all)
}

GCN.layer.link.prediction <- function(input.size,
                                      max.nodes,
                                      batch.size,
                                      num.hidden,
                                      num.filters, 
                                      dropout = 0)
{
  label <- mx.symbol.Variable('label')
  data <- mx.symbol.Variable('data')
  layer.tP <- list()
  layer.outputs <- list()
  K <- length(num.hidden)
  
  for(i in K:1){
    layer.tP[[i]] <- mx.symbol.Variable(paste0("P.",i,".tilde"))
    layer.outputs[[i]] <- Graph.Convolution.batch(data=data,
                                                  tP=layer.tP[[i]],
                                                  num.hidden = num.hidden[i],
                                                  batch.size,
                                                  max.nodes)
  }
  concat.input.size <- input.size + sum(num.hidden)
  graph.conv.stack <- mx.symbol.Concat(data = c(data, layer.outputs), num.args = (K+1), dim = 1)
  conv.1d.input <- mx.symbol.Reshape(mx.symbol.transpose(graph.conv.stack, axes = c(0,1,2)), shape=c(max.nodes, concat.input.size, 1, batch.size))
  
  # 1-D convolution
  # 1st convolutional layer
  concat.input.size <- input.size + sum(num.hidden)
  kernel_1 <- ceiling(concat.input.size/3)
  conv_1 <- mx.symbol.Convolution(data = conv.1d.input, kernel = c(1, kernel_1), num_filter = num.filters[1], pad=c(0,1)) 
  tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh") 
  pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(1,kernel_1), pad=c(0,1)) 
  # 2nd convolutional layer 
  
  kernel_2 <- ceiling(kernel_1/3)
  conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(1, kernel_2), num_filter = num.filters[2], pad=c(0,1)) 
  tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh") 
  pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(1,kernel_2), pad=c(0,1)) 
  
  # Dense layers
  # 1st fully connected layer 
  flatten <- mx.symbol.Flatten(data = pool_2) 
  fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 100) 
  tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh") 
  # 2nd fully connected layer 
  fc_2 <- mx.symbol.FullyConnected(data=tanh_3, num_hidden=2) 
  loss.all <- mx.symbol.SoftmaxOutput(data=fc_2, label=label, name="sm") 
  return(loss.all)
}