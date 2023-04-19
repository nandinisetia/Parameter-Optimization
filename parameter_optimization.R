library(tidyverse)
library(caret)

# load the dataset
url <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
dataset <- read.csv(url, sep=';')
print(dataset)
dataset <- dataset[sample(nrow(dataset)), ]

finalacc<-0
num_samples <- 10
sample_size <- 500

samples <- list()

# loop to generate samples
for (i in 1:num_samples) {
  set.seed(i)
  sample_idx <- createDataPartition(dataset$quality, times=1, p=0.3, list=FALSE)
  sample <- dataset[sample_idx, ]
  train_idx <- createDataPartition(sample$quality, times=1, p=0.7, list=FALSE)
  train_data <- sample[train_idx, -12]
  train_target <- sample[train_idx, 12]
  test_data <- sample[-train_idx, -12]
  test_target <- sample[-train_idx, 12]
  
  # store the training and testing data in a list
  sample_data <- list('train_data'=train_data, 'test_data'=test_data, 
                      'train_target'=train_target, 'test_target'=test_target)
  
  # append the sample data to the samples list
  samples[[i]] <- sample_data
}
print(samples[[1]])
library(kernlab)

# define the kernel list and number of iterations
kernel_list <- c( "vanilladot","rbfdot", "polydot", "tanhdot")
num_iterations <- 1000


cols<-c('Sample','Best Accuracy','Kernel','Nu','Epsilon')
df<-data.frame(matrix(nrow=0,ncol=length(cols)))
colnames(df)=cols

# iterate through each kernel and run the parameter optimization
for(j in 1:10){
  best_accuracy <- 0
  best_kernel <- ""
  best_nu <- 0
  best_epsilon <- 0
  acc <- c()
  iter <- c()
    for (i in 1:num_iterations) {
      # generate random values for nu and epsilon
      nu <- runif(1)
      epsilon <- runif(1)
      kernel=sample(kernel_list,1)
      # randomly select a sample from the samples list
      sample_data <- samples[[j]]
      
      # extract the training and testing data from the sample list
      x_train <- sample_data$train_data
      y_train <- data.frame(sample_data$train_target)
      x_test <- sample_data$test_data
      y_test <- data.frame(sample_data$test_target)
      
      # evaluate the fitness of the current parameters
      model <- ksvm(data.matrix(x_train), y_train, kernel = kernel, nu=nu, epsilon = epsilon, kpar=list())
      
      # make predictions on the test data
      y_pred <- round(predict(model, x_test))
      y_pred=data.frame(y_pred)
      # calculate the accuracy of the model
      accuracy <- round(mean(y_pred == y_test)*100) 
      if(i%%50==0){
        acc<- append(acc, accuracy)
        iter <- append(iter,i)
      }
      # update the best parameters if the current accuracy is better
      if (accuracy > best_accuracy) {
        best_accuracy <- accuracy
        best_kernel <- kernel
        best_nu <- nu
        best_epsilon <- epsilon
      }
    }
  if(best_accuracy>finalacc){
    finalacc=best_accuracy
    xlim <- c(0, 1000)
    ylim <- c(0, 100)
    spacing1<-10
    spacing2<-100
    graph<-plot(NULL, xlim =xlim, ylim =ylim, xlab = "iterations", ylab = "Accuracy", type = "n")
    
    axis(1, at = seq(xlim[1], xlim[2], by = spacing2))
    axis(2, at = seq(ylim[1], ylim[2], by = spacing1))
    title(main = "Fitness Graph")
    lines(iter,acc)
    abline(h = seq(0,100,10), lty = "dashed", col = "gray30")
    
  }
  df[nrow(df)+1,]<-c(j,best_accuracy,best_kernel,best_nu,best_epsilon)
}

print(df)

# Histogram of wine quality
ggplot(dataset, aes(x = quality)) +
  geom_histogram(binwidth = 1, fill = "lightblue", color = "black") +
  labs(title = "Histogram of Wine Quality",
       x = "Quality",
       y = "Frequency") +
  theme(plot.background = element_rect(fill = "lightgray"),
        axis.text = element_text(color = "black"),
        axis.title = element_text(color = "black"))

# Scatter plot of pH vs. alcohol
ggplot(dataset, aes(x = pH, y = alcohol)) +
  geom_point(color = "darkgray", stroke = 1, size = 2) +
  labs(title = "Scatter Plot of pH vs. Alcohol",
       x = "pH",
       y = "Alcohol") +
  theme(plot.background = element_rect(fill = "lightgray"),
        axis.text = element_text(color = "black"),
        axis.title = element_text(color = "black"))
# Box Plot of Residual Sugar by quality
ggplot(dataset, aes(x = quality, y = residual.sugar)) +
  geom_boxplot(fill = "orange", color = "black") +
  labs(title = "Box Plot of Residual Sugar by Wine Quality",
       x = "Quality",
       y = "Residual Sugar") +
  theme(plot.background = element_rect(fill = "lightgray"),
        axis.text = element_text(color = "black"),
        axis.title = element_text(color = "black"))
# Density plot of citric acid by quality
ggplot(dataset, aes(x = citric.acid, fill = factor(quality))) +
  geom_density(alpha = 0.5, color = "black") +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Density Plot of Citric Acid by Quality",
       x = "Citric Acid",
       y = "Density") +
  theme(plot.background = element_rect(fill = "lightgray"),
        axis.text = element_text(color = "black"),
        axis.title = element_text(color = "black"),
        legend.title = element_text(color = "black", size = 12),
        legend.text = element_text(color = "black", size = 10)) +
  guides(fill = guide_legend(reverse = TRUE)) +
  xlim(0, 1.5) +
  ylim(0, 4) +
  scale_x_continuous(breaks = seq(0, 1.5, 0.25), expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0))


# Bar plot of quality counts by alcohol level
dataset %>%
  mutate(alcohol_level = cut(alcohol, breaks = seq(8, 15, by = 1))) %>%
  group_by(alcohol_level, quality) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = alcohol_level, y = count, fill = factor(quality))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Quality Counts by Alcohol Level",
       x = "Alcohol Level",
       y = "Count")




