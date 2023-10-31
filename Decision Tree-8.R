#8 Decision Tree

#Load required libraries
library(rpart)
library(rpart.plot)
library(ggplot2)
library(lattice)
library(caret)


#Subset the data to keep only the relevant columns
sub_data <- subset(accident.df, select = c("Severity","Temperature", "Visibility", "Pressure", "Crossing", "Bump"))

#Convert Severity to a factor variable
sub_data$Severity <- as.factor(sub_data$Severity)

#Handle missing values
sub_data <- na.omit(sub_data)

#Split the data into training and testing sets
set.seed(123)
train_index <- sample(nrow(sub_data), nrow(sub_data) * 0.7)
train_data <- sub_data[train_index, ]
test_data <- sub_data[-train_index, ]

#Define the formula for Decision Tree
formula <- as.formula("Severity ~  Temperature + Visibility + Pressure + Crossing + Bump")

#Train the Decision Tree model
#model <- rpart(formula, data = train_data, control = rpart.control(maxdepth = 5))
model <- rpart(Severity ~ ., data = train_data, method = "class", cp = 0.0039, minsplit = 2, xval = 6)

#Visualize the Decision Tree
rpart.plot(model)

#Make predictions on the test set
pred <- predict(model, newdata = test_data, type = "class")

#Create a confusion matrix
confusion_matrix <- confusionMatrix(data = pred, reference = test_data$Severity)
print(confusion_matrix)

conf_matrix <- table(Actual = test_data$Severity, Predicted = pred)
print(conf_matrix)

#Calculate accuracy
accuracy <- sum(pred == test_data$Severity) / length(pred)
print(paste("Accuracy:", round(accuracy, 4)))

#Accuracy: 87.7%

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

#Generate probabilities for each class
probabilities <- predict(model, newdata = test_data, type = "prob")

#Prepare data for evaluation
labels <- as.factor(test_data$Severity)
class_index <- 4  # Specify the class index you want to evaluate (e.g., 1 for class "1")

#Identify the positive class
positive_class <- levels(labels)[class_index]

#Create a binary response variable indicating positive and negative classes
binary_labels <- ifelse(labels == positive_class, "Positive", "Negative")

#Calculate ROC curve for the specified class
roc <- roc(response = binary_labels, predictor = probabilities[, class_index], levels = c("Positive", "Negative"))

#Plot ROC curve
plot(roc, main = "ROC Curve")

