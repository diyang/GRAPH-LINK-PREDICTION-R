require(Matrix)
require(RSpectra)
require(rjson)
require(igraph)

array2decimal <- function(num.array)
{
  len <- length(num.array)
  for(i in 1:len){
    if(num.array[i] >= 10 && num.array[i]< 100){
      power_i <- i+1
    }else if(num.array[i] >= 100 && num.array[i]< 1000){
      power_i <- i+2
    }else{
      power_i <- i
    }
    
    if(i == 1){
      decimal.num <- num.array[i]*(0.1**power_i)
    }else{
      decimal.num <- decimal.num+num.array[i]*(0.1**power_i)
    }
  }
  return(decimal.num)
}

normalise.adj <- function(adj)
{
  D.sqrt <- sqrt(colSums(adj))
  D.sqrt[is.infinite(D.sqrt)] <- 0
  A.tilde <- adj + diag(dim(adj)[1])
  P <- diag(D.sqrt)%*%A.tilde%*%diag(D.sqrt)
  return(P)
}

chebyshev.polynomials <- function(adj, K)
{
  adj.shape <- dim(adj)
  adj.normalised <- normalise.adj(adj)
  laplacian <- diag(adj.shape[1]) - adj.normalised
  largest_eigval <- eigen(laplacian,only.values = TRUE)$values
  scaled_laplacian <- (2/largest_eigval[1])*laplacian - diag(adj.shape[1])
  t_k <- list()
  t_k[[1]] <- diag(adj.shape[1])
  t_k[[2]] <- scaled_laplacian
  
  hop.deg <- K+1
  if(hop.deg > 2){
    for(i in 3:hop.deg){
      t_k_minus_one <- t_k[[length(t_k)]]
      t_k_minus_two <- t_k[[(length(t_k)-1)]]
      t_k_temp <- 2*scaled_laplacian%*%t_k_minus_one - t_k_minus_two
      t_k[[i]] <- t_k_temp
    }
  }
  return(t_k)
}

subgraph.adj.extract <- function(nodes.pool,
                                 max.nodes = NULL,
                                 adj)
{
  num.nodes <- length(nodes.pool)
  indices.i <- c()
  indices.j <- c()
  edge.weights <- c()
  for(i in 1:num.nodes){
    vi <- nodes.pool[i]
    neighbor.vetices <- which(adj[vi,] > 0)
    selected.neighbor.indices <- which(nodes.pool %in% neighbor.vetices)
    
    if(length(selected.neighbor.indices) > 0){
      for(j in selected.neighbor.indices){
        indices.i <- c(indices.i, i)
        indices.j <- c(indices.j, j)
        vj <- nodes.pool[j]
        edge.weights <- c(edge.weights, adj[vi, vj])
      }
    }
  }
  if(is.null(max.nodes)){
    if(length(edge.weights) > 1){
      sub.adj <- sparseMatrix(x=edge.weights, i=indices.i, j=indices.j, dims=c(num.nodes, num.nodes))
    }else{
      sub.adj <- Matrix(0, num.nodes, num.nodes, sparse = TRUE)
    }
  }else{
    if(length(edge.weights) > 1){
      sub.adj <- sparseMatrix(x=edge.weights, i=indices.i, j=indices.j, dims=c(max.nodes, max.nodes))
    }else{
      sub.adj <- Matrix(0, max.nodes, max.nodes, sparse = TRUE)
    }
  }
  return(sub.adj)
}

Graph.enclose.encode <- function(nodes.pairs,
                                 adj,
                                 K,
                                 max.nodes)
{
  num.pairs <- length(nodes.pairs)
  outputs <- list()
  for(i in 1:num.pairs){
    
    # n hop neigborhood search
    hop <- 1
    is.continue <- TRUE
    all.vertices <- list()
    while(is.continue){
      if(hop == 1){
        temp.nodes.pool <- c(nodes.pairs[[i]]$a, nodes.pairs[[i]]$b)
      }else{
        temp.nodes.pool <- all.vertices[[(hop-1)]]
      }
      
      temp.initial <- TRUE
      for(node in temp.nodes.pool){
        if(temp.initial){
          temp.neigbor <- which(adj[node,]>0)
          temp.initial <- FALSE
        }else{
          temp.neigbor <- c(temp.neigbor, which(adj[node,]>0))
        }
      }
      temp.neigbor <- unique(temp.neigbor)
      
      # check if vertex is already included in the lower hop
      if(hop > 1){
        temp.neigbor <- setdiff(temp.neigbor, c(nodes.pairs[[i]]$a, nodes.pairs[[i]]$b, unlist(all.vertices)))
      }else{
        temp.neigbor <- setdiff(temp.neigbor, c(nodes.pairs[[i]]$a, nodes.pairs[[i]]$b))
      }
      
      if(length(temp.neigbor) == 0){
        is.continue <- FALSE
      }else{
        all.vertices[[hop]] <- temp.neigbor
        hop <- hop+1
        if(length(unlist(all.vertices)) >= (max.nodes-2)){
          is.continue <- FALSE
        }
      }
    }
    
    # Weisfeiler-Lehman Graph labeling
    if(length(all.vertices) > 0){
      for(hop in 1:length(all.vertices)){
        temp.vertices <- all.vertices[[hop]]
        num.temp.vertices <- length(temp.vertices)
        label.vertices <- rep(1, num.temp.vertices)
        decimal.vertices <- rep(1, num.temp.vertices)
        
        is.continue <- TRUE
        while(is.continue){
          for(node.index in 1:num.temp.vertices){
            node <- temp.vertices[node.index]
            neigbor.nodes <- which(adj[node,]>0)
            neigbor.indicies <- which(temp.vertices %in% neigbor.nodes)
            if(length(neigbor.indicies) > 0){
              neigbor.decimals <- array2decimal(sort(label.vertices[neigbor.indicies]))
            }else{
              neigbor.decimals <- 0
            }
            #update labeling
            decimal.vertices[node.index] <- label.vertices[node.index]+neigbor.decimals
          }
          sorted.decimal.vertices <- sort(decimal.vertices, index.return=TRUE)
          
          label <- 1
          label.vertices.old <- label.vertices
          for(sort.index in 1:num.temp.vertices){
            if(sort.index == 1){
              label.vertices[sorted.decimal.vertices$ix[sort.index]] <- label
              tmp.decimal.vertex <- sorted.decimal.vertices$x[sort.index]
            }else{
              if(tmp.decimal.vertex < sorted.decimal.vertices$x[sort.index]){
                label <- label+1
              }
              label.vertices[sorted.decimal.vertices$ix[sort.index]] <- label
              tmp.decimal.vertex <- sorted.decimal.vertices$x[sort.index]
            }
          }
          is.continue <- !(sum(((label.vertices.old-label.vertices)**2)) == 0)
        }
        sorted.vertices <- temp.vertices[order(label.vertices)]
        all.vertices[[hop]] <- sorted.vertices
      }
      all.vertices.flatten <- c(nodes.pairs[[i]]$a, nodes.pairs[[i]]$b, unlist(all.vertices))
      num.vertices <- length(all.vertices.flatten)
      if(num.vertices > max.nodes){
        all.vertices.flatten <- all.vertices.flatten[-((max.nodes+1):num.vertices)]
      }
    }else{
      all.vertices.flatten <- c(nodes.pairs[[i]]$a, nodes.pairs[[i]]$b)
    }
    
    subgraph.adj <- subgraph.adj.extract(all.vertices.flatten, max.nodes, adj)
    subgraph.tP <- chebyshev.polynomials(subgraph.adj, K)
    nodes.pairs.data <- list(a=nodes.pairs[[i]]$a, b=nodes.pairs[[i]]$b, sorted_neighbors=all.vertices.flatten, tP=subgraph.tP, adj=subgraph.adj)
    outputs[[i]] <- nodes.pairs.data
  }
  return(outputs)
}

Graph.receptive.fields.computation <- function(nodes.pool,
                                               P, 
                                               adj,
                                               random.neighbor)
{
  layer.tP <- list()
  nodes.sort <- sort(nodes.pool)
  layer.H <- list(nodes.sort)
  included.node <- layer.H[[1]]
  K <- length(random.neighbor)
  node.size <- length(nodes.pool)
  
  for(ly in 1:K){
    # determin the dimensions of P matrix
    if(ly == 1){
      dim1 <- node.size
    }else{
      dim1 <- dim2
    }
    dim2 <- dim1*(random.neighbor[ly])
    P.tilde <- matrix(0, dim1, dim2)
    
    # define container
    all.vectices <- c()
    P.neighbor <- list()
    
    # determine neighborhood
    count <- 1
    for(node in layer.H[[ly]]){
      neighbor.index  <- which(adj[node,] > 0)
      neighbor.number <- length(neighbor.index)
      neighbor.index.diff  <- setdiff(neighbor.index, included.node)
      neighbor.diff.number <- length(neighbor.index.diff)
      
      #neighbor.index  <- setdiff(neighbor.index, c(node))
      if(neighbor.diff.number == 0){
        neighbor.vec <- NULL
      }else{
        if(neighbor.diff.number <= (random.neighbor[ly])){
          neighbor.vec <- neighbor.index.diff
        }else{
          sample.neighbor.index <- sample(neighbor.diff.number, random.neighbor[ly], replace=FALSE)
          neighbor.vec <- neighbor.index.diff[sample.neighbor.index]
        }
        all.vectices <- c(all.vectices, neighbor.vec)
      }
      node.stats <- list(node_index = node, neighbor_num = neighbor.number, neighbor_vec = neighbor.vec)
      P.neighbor[[count]] <- node.stats
      count <- count + 1
    }  
    sort.unique.vectices <- sort(unique(all.vectices))
    included.node <- c(included.node, sort.unique.vectices)
    
    # determine P matrix
    for(ti in 1:length(P.neighbor)){
      node.info <- P.neighbor[[ti]]
      if(!is.null(node.info$neighbor_vec)){
        node.neighbor.indcies <- match(node.info$neighbor_vec, sort.unique.vectices)
        for(j in 1:length(node.info$neighbor_vec)){
          vi <- node.info$node_index
          vj <- node.info$neighbor_vec[j]
          tj <- node.neighbor.indcies[j]
          if(length(node.info$neighbor_vec) <= random.neighbor[ly]){
            d <- length(node.info$neighbor_vec)
          }else{
            d <- random.neighbor[ly]
          }
          P.tilde[ti,tj] <- P[vi,vj]*(node.info$neighbor_num/d)
        }
      }
    }
    layer.tP[[ly]] <-P.tilde
    layer.H[[ly+1]] <- sort.unique.vectices
  }
  outputs <- list(tP=layer.tP, H=layer.H)
  return(outputs)
}

loaddata.ppi <- function(){
  json_graph <- "I:/Desktop/R/SAGE-GRAPH-R/example_data/toy-ppi-G.json"
  #json_graph <- "./example_data/PPI/toy-ppi-G.json"
  G_data  <- fromJSON(paste(readLines(json_graph), collapse=""))
  edges <- matrix(unlist(G_data$links), ncol = 4, byrow = TRUE)[,3:4]+1
  graph <- graph_from_edgelist(edges, directed = FALSE)
  adjmatrix <- as_adj(graph, type = 'both', sparse = igraph_opt("sparsematrices"))
  #graph2 <- graph_from_adjacency_matrix(adjmatrix, mode='undirected', diag=FALSE)
  
  json_class <- "I:/Desktop/R/SAGE-GRAPH-R/example_data/toy-ppi-class_map.json"
  #json_class <- "./example_data/PPI/toy-ppi-class_map.json"
  G_class  <- fromJSON(paste(readLines(json_class), collapse=""))
  
  json_idmap <- "I:/Desktop/R/SAGE-GRAPH-R/example_data/toy-ppi-id_map.json"
  #json_idmap <- "./example_data/PPI/toy-ppi-id_map.json"
  G_idmap  <- fromJSON(paste(readLines(json_idmap), collapse=""))
  
  csv_feats <- "I:/Desktop/R/SAGE-GRAPH-R/example_data/toy-ppi-feats.csv"
  #csv_feats <- "./example_data/PPI/toy-ppi-feats.csv"
  feats <- read.csv(csv_feats, header = FALSE)
  
  D.sqrt <- sqrt(colSums(adjmatrix))
  
  A.tilde <- adjmatrix + Diagonal(dim(adjmatrix)[1])
  
  P <- diag(D.sqrt)%*%A.tilde%*%diag(D.sqrt)
  
  outputs <- list(adjmatrix = adjmatrix, P = P, Atilde = A.tilde, Dsqrt = D.sqrt, features = feats, graph = graph, class = G_class)
  return(outputs)
}

loaddata.cora <- function(){
  csv_cites <- "./example_data/CORA/cites.csv"
  edges.cites <- read.csv(csv_cites, header = FALSE)
  edges.cites <- as.matrix(edges.cites[2:dim(edges.cites)[1],])
  
  csv_paper <- "./example_data/CORA/paper.csv"
  paper.class <- read.csv(csv_paper, header = FALSE)
  paper.class <- as.matrix(paper.class[2:dim(paper.class)[1],])
  
  csv_content <- "./example_data/CORA/content.csv"
  content.class <- read.csv(csv_content, header = FALSE)
  content.class <- content.class[2:dim(content.class)[1],]
  column.names <-  c("paper_id",as.character(unique(content.class$V2)),"class")
  num.cols <- length(column.names)
  num.paper <- dim(paper.class)[1]
  
  content.df <- data.frame(matrix(0, ncol = num.cols, nrow = num.paper))
  colnames(content.df) <- column.names
  
  for(i in 1:dim(paper.class)[1]){
    inds.v1 <- which(edges.cites[,1] == paper.class[i,1])
    inds.v2 <- which(edges.cites[,2] == paper.class[i,1])
    
    edges.cites[inds.v1,1] <- i
    edges.cites[inds.v2,2] <- i
    
    inds.paper_id <-which(content.class[,1] == paper.class[i,1])
    cite.ids <- content.class[inds.paper_id,2]
    content.df[i,1] <- paper.class[i,1]
    content.df[i,num.cols] <- paper.class[i,2]
    for(j in 1:length(cite.ids)){
      ind.class <- which(column.names == cite.ids[j])
      content.df[i,ind.class] <- 1 
    }
  }
  edges.cites<-apply(edges.cites, 2, as.numeric)
  class(edges.cites) <- "numeric"
  storage.mode(edges.cites) <- "numeric"
  
  graph <- graph_from_edgelist(edges.cites, directed = FALSE)
  adjmatrix <- as_adj(graph, type = 'both', sparse = igraph_opt("sparsematrices"))
  
  D.sqrt <- sqrt(colSums(adjmatrix))
  
  A.tilde <- adjmatrix + Diagonal(dim(adjmatrix)[1])
  
  P <- diag(D.sqrt)%*%A.tilde%*% diag(D.sqrt)
  
  outputs <- list(adjmatrix = adjmatrix, P = P, Atilde = A.tilde, Dsqrt = D.sqrt, graph = graph, content = content.df)
  return(outputs)
}

sub.graph.display <- function(sub.graph.input)
{
  max.nodes <- dim(sub.graph.input$adj)[1]
  sub.graph <- graph_from_adjacency_matrix(sub.graph.input$adj, mode = "undirected" )
  num.vertices <- length(sub.graph.input$sorted_neighbors)
  if(num.vertices < max.nodes){
    padding.vertices <- c(-1:-(max.nodes-num.vertices))
    vertices.names <- c(sub.graph.input$sorted_neighbors, padding.vertices)
  }else{
    vertices.names <- sub.graph.input$sorted_neighbors
  }
  
  V(sub.graph)$name <- vertices.names
  iso <- which(V(sub.graph)$name < 0)
  sub.graph <- delete.vertices(sub.graph, iso)
  
  a <- sub.graph.input$a
  b <- sub.graph.input$b
  
  plot(sub.graph, vertex.label.color="black",
       vertex.color=c( "tomato", "gold")[1+(V(sub.graph)$name %in% c(a,b))],
       edge.width=3, vertex.size = c(15,18)[1+(V(sub.graph)$name %in% c(a,b))])
}