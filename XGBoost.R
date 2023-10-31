#6 XGBoost

library(xgboost)
library(caret)

sub_data <- subset(accident.df, select = c("Severity", "Wind_Speed", "Temperature", "Visibility", "Pressure", "Month", "Year", "Traffic_Signal", "Bump", "Crossing"))
sub_data <- na.omit(sub_data)

sub_data$Severity <- as.factor(sub_data$Severity)


set.seed(123)
train_indices <- sample(1:nrow(sub_data), nrow(sub_data) * 0.7)  # 70% for training
train_data <- sub_data[train_indices, ]
test_data <- sub_data[-train_indices, ]

x_train <- as.matrix(train_data[, !(names(train_data) %in% "Severity")])
y_train <- as.numeric(train_data$Severity)

x_test <- as.matrix(test_data[, !(names(test_data) %in% "Severity")])
y_test <- as.numeric(test_data$Severity)

xgb_model <- xgboost(data = x_train, label = y_train, objective = "multi:softmax", num_class = 4, nrounds = 100)

predictions <- predict(xgb_model, x_test)

accuracy <- sum(predictions == y_test) / length(y_test)
print(paste("Accuracy:", accuracy))

#Accuracy: 92.29%

# Convert predictions and actual labels to factors
predictions <- as.factor(predictions)
y_test <- as.factor(y_test)

# Create the confusion matrix
conf_matrix <- table(Actual = y_test, Predicted = predictions)
print(conf_matrix)

precision <- diag(conf_matrix) / colSums(conf_matrix)
recall <- diag(conf_matrix) / rowSums(conf_matrix)
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print precision, recall, and F1 score
print(paste("Precision:", round(precision, 4)))
print(paste("Recall:", round(recall, 4)))
print(paste("F1 Score:", round(f1_score, 4)))

# Create a data frame for the evaluation metrics
metrics <- data.frame(
  Metric = c("Recall", "Precision", "F1 Score"),
  Value = c(recall, precision, f1_score)
)

# Plot the evaluation metrics
barplot(metrics$Value, names.arg = metrics$Metric,
        col = "darkblue", main = "Evaluation Metrics",
        xlab = "Metric", ylab = "Value")
