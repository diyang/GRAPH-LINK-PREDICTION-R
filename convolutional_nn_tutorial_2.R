# This script is used to resize images from 64x64 to 28x28 pixels

# Clear workspace
rm(list=ls())

# Load EBImage library
require(EBImage)

# Load data
X <- read.csv("./example_data/MINST/olivetti_X.csv", header = F)
labels <- read.csv("./example_data/MINST/olivetti_y.csv", header = F)

# Dataframe of resized images
rs_df <- data.frame()

# Main loop: for each image, resize and set it to greyscale
for(i in 1:nrow(X))
{
  # Try-catch
  result <- tryCatch({
    # Image (as 1d vector)
    img <- as.numeric(X[i,])
    # Reshape as a 64x64 image (EBImage object)
    img <- Image(img, dim=c(64, 64), colormode = "Grayscale")
    # Resize image to 28x28 pixels
    img_resized <- resize(img, w = 28, h = 28)
    # Get image matrix (there should be another function to do this faster and more neatly!)
    img_matrix <- img_resized@.Data
    # Coerce to a vector
    img_vector <- as.vector(t(img_matrix))
    # Add label
    label <- labels[i,]
    vec <- c(label, img_vector)
    # Stack in rs_df using rbind
    rs_df <- rbind(rs_df, vec)
    # Print status
    print(paste("Done",i,sep = " "))},
    # Error function (just prints the error). Btw you should get no errors!
    error = function(e){print(e)})
}


# Set names. The first columns are the labels, the other columns are the pixels.
names(rs_df) <- c("label", paste("pixel", c(1:784)))

# Train-test split
#-------------------------------------------------------------------------------
# Simple train-test split. No crossvalidation is done in this tutorial.

# Set seed for reproducibility purposes
set.seed(100)

# Shuffled df
shuffled <- rs_df[sample(1:400),]

# Train-test split
train_28 <- shuffled[1:360, ]
test_28 <- shuffled[361:400, ]

# Save train-test datasets
write.csv(train_28, "./example_data/MINST/train_28.csv", row.names = FALSE)
write.csv(test_28, "./example_data/MINST/test_28.csv", row.names = FALSE)

# Done!
print("Done!")