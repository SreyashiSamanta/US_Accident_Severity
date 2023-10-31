library(class)
library(mice)
library(caret)

#Subset the data to keep only the relevant columns
sub_data <- subset(accident.df, select = c("Severity", "Wind_Speed", "Temperature", "Visibility", "Pressure", "Month", "Year", "Traffic_Signal", "Bump", "Crossing"))

#Convert Severity to a factor variable
sub_data$Severity <- as.factor(sub_data$Severity)

#Impute missing values
sub_data_imputed <- mice(sub_data)

#Extract completed data from the imputed object
completed_data <- complete(sub_data_imputed)

#Convert input variables to numeric
completed_data[, -1] <- lapply(completed_data[, -1], as.numeric)

#Split the data into training and testing sets
set.seed(123)
train_index <- sample(nrow(completed_data), nrow(completed_data) * 0.7)
train_data <- completed_data[train_index, ]
test_data <- completed_data[-train_index, ]

#Define the formula for KNN
formula <- as.formula("Severity ~ Wind_Speed + Temperature + Visibility + Pressure + Month + Year + Traffic_Signal + Bump + Crossing")

#Train the KNN model with different k values
k_values <- 1:20  # Adjust the list of k values as needed

accuracy_matrix <- matrix(0, nrow = length(k_values), ncol = 2)

for (i in 1:length(k_values)) {
  k <- k_values[i]
  
  #Train the KNN model
  model <- knn(train_data[, -1], test_data[, -1], train_data$Severity, k)
  
  #Make predictions on the test set
  pred <- as.factor(model)
  
  #Calculate accuracy
  accuracy <- sum(pred == test_data$Severity) / length(test_data$Severity)
  
  #Update the accuracy matrix
  accuracy_matrix[i, ] <- c(k, accuracy)
}

#Print the accuracy matrix
colnames(accuracy_matrix) <- c("k", "Accuracy")
print(accuracy_matrix)

#Select the best value of k based on the highest accuracy
best_k <- accuracy_matrix[which.max(accuracy_matrix[, "Accuracy"]), "k"]
print(paste("Best k value:", best_k))

#Train the final KNN model with k=9
final_model <- train(
  formula,
  data = train_data,
  method = "knn",
  tuneGrid = data.frame(k = best_k),
  trControl = trainControl(method = "cv", number = 5)
)


#Make predictions on the test set
pred <- predict(final_model, newdata = test_data)

#Calculate Accuracy
accuracy <- sum(pred == test_data$Severity) / nrow(test_data)
print(paste("Accuracy:", round(accuracy, 4)))

#Accuracy: 93.22%

#Create a confusion matrix
confusion <- confusionMatrix(pred, test_data$Severity)
print(confusion)

conf_matrix <- table(Actual = test_data$Severity, Predicted = pred)
print(conf_matrix)


#Plot the confusion matrix
heatmap(conf_matrix, col = c("darkgreen", "darkred"),
        main = "Confusion Matrix",
        xlab = "Predicted", ylab = "Actual")


#Calculate precision, recall, and F1 score
precision <- diag(conf_matrix) / colSums(conf_matrix)
recall <- diag(conf_matrix) / rowSums(conf_matrix)
f1_score <- 2 * (precision * recall) / (precision + recall)

#Print precision, recall, and F1 score
print(paste("Precision:", round(precision, 4)))
print(paste("Recall:", round(recall, 4)))
print(paste("F1 Score:", round(f1_score, 4)))

#Create a data frame for the evaluation metrics
metrics <- data.frame(
  Metric = c("Recall", "Precision", "F1 Score"),
  Value = c(recall, precision, f1_score)
)

#Plot the evaluation metrics
barplot(metrics$Value, names.arg = metrics$Metric,
        col = "darkblue", main = "Evaluation Metrics",
        xlab = "Metric", ylab = "Value")

