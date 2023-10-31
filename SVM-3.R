#3 Support Vector Machines (SVM)

# Load required libraries
library(e1071)

# Subset the data to keep only the relevant columns
sub_data <- subset(accident.df, select = c("Severity", "Wind_Speed", "Temperature", "Visibility", "Pressure"))
# Drop rows with missing values
sub_data <- na.omit(sub_data)

# Convert Severity to a factor variable
sub_data$Severity <- as.factor(sub_data$Severity)

# Split the data into training and testing sets
set.seed(123)
train_index <- sample(nrow(sub_data), nrow(sub_data) * 0.7)
train_data <- sub_data[train_index, ]
test_data <- sub_data[-train_index, ]

# Train the SVM model
model <- svm(Severity ~ ., data = train_data)

# Make predictions on the test set
pred <- predict(model, newdata = test_data)

# Calculate accuracy
accuracy <- sum(pred == test_data$Severity) / nrow(test_data)
print(paste("Accuracy:", round(accuracy, 4)))

#Accuracy 93.22%

# Create a confusion matrix
confusion <- confusionMatrix(pred, test_data$Severity)
print(confusion)

conf_matrix <- table(pred, test_data$Severity)
print(conf_matrix)

# Plot the confusion matrix
heatmap(conf_matrix, col = c("darkgreen", "darkred"),
        main = "Confusion Matrix",
        xlab = "Predicted", ylab = "Actual")

# Calculate precision, recall, and F1 score
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

