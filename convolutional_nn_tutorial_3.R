# Clean workspace
rm(list=ls())

# Load MXNet
require(mxnet)
setwd("~/Documents/GRAPH-LINK-PREDICTION-R")
source("train.R")

alexnet.sym <- function()
{
  data <- mx.symbol.Variable('data')
  label <- mx.symbol.Variable('label')
  # 1st convolutional layer
  conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
  tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
  pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  # 2nd convolutional layer
  conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
  tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
  pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  # 1st fully connected layer
  flatten <- mx.symbol.Flatten(data = pool_2)
  fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
  tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
  # 2nd fully connected layer
  fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)
  # Output. Softmax output since we'd like to get some probabilities.
  NN_model <- mx.symbol.SoftmaxOutput(data = fc_2, label= label, name="sm")
  return(NN_model)
}

alexnet.setup <- function(alexnet.sym, 
                          batch.size,
                          ctx = mx.ctx.default(),
                          initializer=mx.init.uniform(0.01))
{
  arg.names <- alexnet.sym$arguments
  input.shape <- list()
  for(name in arg.names){
    if( grepl('label$', name) )
    {
      input.shape[[name]] <- c(batch.size)
    }else if( grepl('data$', name) ){
      input.shape[[name]] <- c(28,28,1,batch.size)
    }
  }
  
  params <- mx.model.init.params(symbol = alexnet.sym, input.shape = input.shape, initializer = initializer, ctx = ctx)
  
  args <- input.shape
  args$symbol <- alexnet.sym
  args$ctx <- ctx
  args$grad.req <- 'write'
  alexnet.exec <- do.call(mx.simple.bind, args)
  
  mx.exec.update.arg.arrays(alexnet.exec, params$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(alexnet.exec, params$aux.params, match.name = TRUE)
  
  #grad.arrays <- list()
  #for (name in names(alexnet.exec$ref.grad.arrays)) {
  #  if (is.param.name(name))
  #    grad.arrays[[name]] <- alexnet.exec$ref.arg.arrays[[name]]*0
  #}
  #mx.exec.update.grad.arrays(alexnet.exec, grad.arrays, match.name=TRUE)
  
  return (list(alexnet.exec = alexnet.exec, 
               symbol = alexnet.sym,
               params = params,
               batch.size = batch.size))
}

alexnet.train <- function(model,
                          train_array,
                          train_y,
                          num.epoch = 480,
                          learning.rate = 0.01,
                          momentum = 0.9,
                          weight.decay = 0,
                          clip.gradient = 1,
                          optimizer = 'sgd',
                          lr.scheduler = NULL)
{
  m <- model
  batch.size <- m$batch.size
  
  opt <- mx.opt.create(optimizer, learning.rate = learning.rate,
                       wd = weight.decay,
                       rescale.grad = 1/batch.size,
                       clip_gradient=clip.gradient,
                       momentum = momentum,
                       lr_scheduler = lr.scheduler)
  
  opt.updater <- mx.opt.get.updater(opt, m$alexnet.exec$ref.arg.arrays)
  
  
  
  for(epoch in 1:num.epoch){
    cat(paste0('Training Epoch ', epoch, '\n'))
    
    ##################
    # batch training #
    ##################
    train.nll <- 0
    num.batch.train <- floor(length(train_y)/batch.size)
    train.metric <- mx.metric.accuracy$init()
    for(batch.counter in 1:num.batch.train){
      # gcn input data preparation
      batch.begin <- (batch.counter-1)*batch.size+1
      
      train.data  <- train_array[,,1,batch.begin:(batch.begin+batch.size-1)]
      dim(train.data) <- c(28, 28, 1, batch.size)
      train.label <- train_y[batch.begin:(batch.begin+batch.size-1)]
      
      alexnet.input <- list()
      alexnet.input[['data']]  <- mx.nd.array(train.data)
      alexnet.input[["label"]] <- mx.nd.array(train.label)
      
      mx.exec.update.arg.arrays(m$alexnet.exec, alexnet.input, match.name = TRUE)
      mx.exec.forward(m$alexnet.exec, is.train = TRUE)
      mx.exec.backward(m$alexnet.exec)
      arg.blocks <- opt.updater(m$alexnet.exec$ref.arg.arrays, m$alexnet.exec$ref.grad.arrays)
      mx.exec.update.arg.arrays(m$alexnet.exec, arg.blocks, skip.null=TRUE)
      
      label.probs <- mx.nd.choose.element.0index(m$alexnet.exec$ref.outputs[["sm_output"]], m$alexnet.exec$ref.arg.arrays[["label"]])
      train.nll <- train.nll + calc.nll(as.array(label.probs), batch.size)
      
      train.metric <- mx.metric.accuracy$update(m$alexnet.exec$ref.arg.arrays[["label"]], m$alexnet.exec$ref.outputs[["sm_output"]], train.metric)
      
      #cat(paste0("Epoch [", epoch, "] Batch [", batch.counter, "] Trian: NLL=", train.nll / batch.counter,"\n"))
    }
    result <- mx.metric.accuracy$get(train.metric)
    cat(paste0("[", epoch, "] Train-", result$name, "=", result$value, "\n"))
  }
  model <- mx.model.extract.model(m$symbol, m$alexnet.exec)
}

# Loading data and set up
#-------------------------------------------------------------------------------

# Load train and test datasets
train <- read.csv("./example_data/MINST/train_28.csv")
test <- read.csv("./example_data/MINST/test_28.csv")

# Set up train and test datasets
train <- data.matrix(train)
train_x <- t(train[, -1])
train_y <- train[, 1]
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))

test_x <- t(test[, -1])
test_y <- test[, 1]
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x))

mx.set.seed(100)
alexnet_sym <- alexnet.sym()
alexnet_model <- alexnet.setup(alexnet_sym, 40)
trained_alexnet <- alexnet.train(alexnet_model, num.epoch=500, train_array, train_y)









# Set up the symbolic model
#-------------------------------------------------------------------------------

data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

# Device used. CPU in my case.
devices <- mx.cpu()

# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 480,
                                     array.batch.size = 40,
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Testing
#-------------------------------------------------------------------------------

# Predict labels
predicted <- predict(model, test_array)
# Assign labels
predicted_labels <- max.col(t(predicted)) - 1
# Get accuracy
sum(diag(table(test[, 1], predicted_labels)))/40

################################################################################
#                           OUTPUT
################################################################################
#
# 0.975
#