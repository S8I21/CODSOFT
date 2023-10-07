install.packages("tidyverse")
install.packages("caret")
install.packages("dplyr")
library(tidyverse)
library(caret)
write.csv(titanic_data, "titanic_dataset.csv", row.names = FALSE)
titanic_data <- read.csv("titanic_dataset.csv", stringsAsFactors = TRUE)
titanic <- as.data.frame(titanic_data)
head(titanic)
summary(titanic)
str(titanic)
# Handle missing values
titanic$Age[is.na(titanic$Age)] <- median(titanic$Age, na.rm = TRUE)
titanic$Embarked[is.na(titanic$Embarked)] <- "S"

# Convert categorical variables to factors
titanic$Survived <- as.factor(titanic$Survived)
titanic$Pclass <- as.factor(titanic$Pclass)
titanic$Sex <- as.factor(titanic$Sex)
titanic$Embarked <- as.factor(titanic$Embarked)

# Split the data into training and testing sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(titanic_data$Survived, p = 0.7, list = FALSE)
train_data <- titanic[train_index, ]
test_data <- titanic[-train_index, ]

#Build and train learning machine model
# Specify starting values for coefficients
start_values <- c("(Intercept)" = 0, "Pclass2" = 0, "Pclass3" = 0, "Sexmale" = 0, "Age" = 0, 
                  "SibSp" = 0, "Parch" = 0, "Fare" = 0, "EmbarkedQ" = 0, "EmbarkedS" = 0)

# Fit the logistic regression model with starting values
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = train_data, family = binomial(link = "logit"), maxit = 1000)

# Check for convergence
if (model$converged) {
  # Model converged without issues
  summary(model)
} else {
  # Model did not converge
  cat("Warning: Model did not converge\n")
  # Consider additional troubleshooting steps or modifying the model
}

#Make Predictions

predictions <- predict(model, newdata = test_data, type = "response")

#Evaluate the model

confusion_matrix <- table(Actual = test_data$Survived, Predicted = ifelse(predictions > 0.5, 1, 0))
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")

train_data <- na.omit(train_data)

# Impute missing values in Age with median
train_data$Age[is.na(train_data$Age)] <- median(train_data$Age, na.rm = TRUE)

# Impute missing values in Embarked with mode
embarked_mode <- names(sort(table(train_data$Embarked), decreasing = TRUE))[1]
train_data$Embarked[is.na(train_data$Embarked)] <- embarked_mode

# Create a control function for cross-validation
ctrl <- trainControl(method = "cv", number = 5)

# Perform grid search with cross-validation for logistic regression
tune_model <- train(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_data,
  method = "glm",
  trControl = ctrl
)

# View the best hyperparameters (not applicable to logistic regression)
best_hyperparameters <- tune_model$bestTune
print(best_hyperparameters)

# Assuming you have trained the model and tuned hyperparameters (if necessary)

# Assuming you have trained the model and tuned hyperparameters (if necessary)

# Train the final model using the best hyperparameters
final_model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_data,
  family = binomial(link = "logit")  # Use the logit link function for logistic regression
)

# Now you can use the final_model to make predictions on new data
new_data <- data.frame(
  Pclass = c(1, 3),
  Sex = factor(c("female", "male")),
  Age = c(25, 30),
  SibSp = c(1, 0),
  Parch = c(2, 0),
  Fare = c(100, 7.5),
  Embarked = factor(c("S", "C"))
)


# Sample training data (replace with your actual training data)
train_data <- data.frame(
  Survived = c(1, 0, 1, 0, 1),
  Pclass = factor(c(1, 2, 3, 1, 3)),  # Convert Pclass to a factor
  Sex = factor(c("female", "male", "female", "male", "female")),
  Age = c(25, 30, 22, 40, 28),
  SibSp = c(1, 0, 1, 0, 1),
  Parch = c(2, 0, 1, 0, 2),
  Fare = c(100, 7.5, 15, 80, 10),
  Embarked = factor(c("S", "C", "S", "S", "C")) 
)

# Sample new data for predictions (replace with your actual data)
new_data <- data.frame(
  Pclass = factor(c(1, 3, 2, 1, 3, 2, 1, 3, 2, 1)),  # Pclass as a factor
  Sex = factor(c("female", "male", "female", "male", "female", "male", "female", "male", "female", "male")),
  Age = c(25, 30, 22, 40, 28, 35, 18, 45, 29, 38),
  SibSp = c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
  Parch = c(2, 0, 1, 0, 2, 1, 0, 3, 2, 1),
  Fare = c(100, 7.5, 15, 80, 10, 25, 150, 8, 12, 60),
  Embarked = factor(c("S", "C", "S", "S", "C", "S", "S", "C", "S", "C"))
)


# Train the logistic regression model
final_model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_data,
  family = binomial(link = "logit")
)

# Make predictions
new_predictions <- predict(final_model, newdata = new_data, type = "response")

# Set a probability threshold (e.g., 0.5) for binary predictions
threshold <- 0.5

# Make binary predictions based on the threshold
binary_predictions <- ifelse(new_predictions >= threshold, "Survived", "Did Not Survive")

# Display the results
results <- data.frame(
  Passenger = 1:2,
  Predicted_Survival = binary_predictions,
  Predicted_Probability = new_predictions
)

print(results)
# Load necessary libraries
library(ggplot2)

# Create a subset of the data for visualization
subset_data <- subset(titanic, select = c(Survived, Pclass, Sex, Embarked))

# Create a bar plot to compare survival by Pclass
ggplot(data = subset_data, aes(x = Pclass, fill = Survived)) +
  geom_bar() +
  labs(title = "Comparison of Survival by Pclass(0=Not survive,1=survive)",
       x = "Pclass", y = "Count") +
  theme_minimal()

# Create a bar plot to compare survival by Sex
ggplot(data = subset_data, aes(x = Sex, fill = Survived)) +
  geom_bar() +
  labs(title = "Comparison of Survival by Sex(0<-Not survive,1<-survive)",
       x = "Sex", y = "Count") +
  theme_minimal()

# Create a bar plot to compare survival by Embarked
ggplot(data = subset_data, aes(x = Embarked, fill = Survived)) +
  geom_bar() +
  labs(title = "Comparison of Survival by Embarked(0<-Not survive,1<-survive)",
       x = "Embarked", y = "Count") +
  theme_minimal()
