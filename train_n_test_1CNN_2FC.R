# Clean workspace
rm(list=ls())

# Setting the current working directory to be the folder containing the files
setwd("E:/Datasets/TrafficLight/MXNet Training")

# Load MXNet
require(mxnet)

# Loading data and set up
#-------------------------------------------------------------------------------

# Load train and test datasets
train <- read.csv("train_28.csv")
test <- read.csv("test_28.csv")

# Set up train and test datasets
train <- data.matrix(train)
train_x <- t(train[, -1])
train_y <- train[, 1]
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x)) # redefine the dimension of train array

test_x <- t(test[, -1])
test_y <- test[, 1]
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x)) # redefine the dimension of test array

# Set up the symbolic model
#-------------------------------------------------------------------------------

data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
## 2nd convolutional layer
#conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
#tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
#pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_1)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 7)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
#mx.set.seed(100)

# Device used. CPU in my case.
devices <- mx.cpu()

# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 300,
                                     array.batch.size = 40,
                                     learning.rate = 0.05,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Testing
#-------------------------------------------------------------------------------

# Predict labels
predicted <- predict(model, test_array)
# Assign labels
predicted_labels <- max.col(t(predicted)) -1  # somehow Mic got this nasty -1 throughout the calculation. Need to make it right. In the corrected_probability vector I also catered to this -1.
predicted_labels
# Get accuracy
# sum(diag(table(test[, 1], predicted_labels)))/40

# Confusion matrix

# Test Accuracy
1- mean(predicted_labels != test[, 1])
test[,1]

# Print the vector that shows the accuracy of each prediction
correct_probability <- data.frame()
# correcting the mess created by -1 in the predicted_label
corrected_test <- 1:40
for(i in 1:nrow(test))
{
  if (test[i,1] <7)
    {
    corrected_test[i] <- test[i,1] + 1
    } else 
    {corrected_test[i] <- 1
    }
}


for(i in 1:nrow(test))
{
  vec <- c(test[i,1],predicted_labels[i],predicted[corrected_test[i],i], predicted[predicted_labels[i]+1,i])
  correct_probability <- rbind(correct_probability, vec)
}
names(correct_probability) <- c("true","predicted","guess_correct_prob","highest_prob")
correct_probability
write.csv(correct_probability, "E:/Datasets/TrafficLight/MXNet Training/test_28_probability.csv", row.names = FALSE)

################################################################################
#                           OUTPUT
################################################################################
#
# 0.975
#

