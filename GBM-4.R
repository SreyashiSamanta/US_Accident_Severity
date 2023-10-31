#4 Gradient Boosting Machine (GBM)

#Load required libraries
library(gbm)
library(dplyr)

#Subset the data to keep only the relevant columns
sub_data <- subset(accident.df, select = c("Severity", "Wind_Speed", "Temperature", "Visibility","Month", "Pressure","Year"))

#Drop rows with missing values
sub_data <- na.omit(sub_data)

#Convert Severity to a factor variable
sub_data$Severity <- factor(sub_data$Severity, levels = c(1, 2, 3, 4),
                            labels = c("Fatal", "Injury", "Property Damage Only", "Unknown Severity"))

#Split the data into training and testing sets
set.seed(123)
train_index <- sample(nrow(sub_data), nrow(sub_data) * 0.7)
train_data <- sub_data[train_index, ]
test_data <- sub_data[-train_index, ]

models <- lapply(levels(train_data$Severity), function(level) {
  train_data_temp <- train_data
  train_data_temp$Severity <- as.integer(train_data_temp$Severity == level)
  
  gbm_model <- gbm(Severity ~ ., data = train_data_temp, distribution = "bernoulli", n.trees = 100, interaction.depth = 3, shrinkage = 0.1, verbose = FALSE)
  
  list(level = level, model = gbm_model)
})

#Make predictions on the test set for each model
pred_probs <- lapply(models, function(model) {
  test_data_temp <- test_data
  test_data_temp$Severity <- as.integer(test_data_temp$Severity == model$level)
  
  predict(model$model, newdata = test_data_temp, n.trees = 100, type = "response")
})

#Combine the predictions for all classes
pred <- apply(do.call(cbind, pred_probs), 1, function(row_probs) {
  levels(train_data$Severity)[which.max(row_probs)]
})

#Calculate accuracy
accuracy <- sum(pred == test_data$Severity) / nrow(test_data)
print(paste("Accuracy:", round(accuracy, 4)))

#Accuracy: 92.93%

#Generate confusion matrix
conf_mat <- table(test_data$Severity, pred)
print(conf_mat)


#Plot the confusion matrix
heatmap(conf_mat, col = c("darkgreen", "darkred"),
        xlab = "Predicted", ylab = "Actual")

#Calculate recall and precision for each class
recall <- diag(conf_mat) / rowSums(conf_mat)
precision <- diag(conf_mat) / colSums(conf_mat)
f1_score <- 2 * (precision * recall) / (precision + recall)

print(paste("Recall:", round(recall, 4)))
print(paste("Precision:", round(precision, 4)))
print(paste("F1 Score:", round(f1_score, 4)))

