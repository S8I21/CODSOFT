install.packages("ggplot2")
install.packages("corrplot")
install.packages("caret")
install.packages("e1071")
library("ggplot2")
library("corrplot")
library("caret")
library("e1071")
write.csv(iris, "IRIS.csv", row.names = FALSE)
Iris<- read.csv("IRIS.csv", stringsAsFactors = TRUE)
Iris1 <- as.data.frame(Iris)
summary(Iris1)
# Step 4: Data Visualization
# Pairwise scatterplot matrix
pairs(iris[, 1:4], col = iris$Species, pch = 19)

# Boxplots for each attribute by species
ggplot(iris, aes(x = Species, y = Sepal.Length)) + geom_boxplot()
ggplot(iris, aes(x = Species, y = Sepal.Width)) + geom_boxplot()
ggplot(iris, aes(x = Species, y = Petal.Length)) + geom_boxplot()
ggplot(iris, aes(x = Species, y = Petal.Width)) + geom_boxplot()

# Step 5: Feature Distribution
# Histograms by species for each attribute
ggplot(iris, aes(x = Sepal.Length, fill = Species)) + geom_histogram(binwidth = 0.2)
ggplot(iris, aes(x = Sepal.Width, fill = Species)) + geom_histogram(binwidth = 0.2)
ggplot(iris, aes(x = Petal.Length, fill = Species)) + geom_histogram(binwidth = 0.2)
ggplot(iris, aes(x = Petal.Width, fill = Species)) + geom_histogram(binwidth = 0.2)

# Step 6: Correlation Analysis
correlation_matrix <- cor(iris[, 1:4])
corrplot(correlation_matrix, method = "color")

# Step 7: Predictive Modeling
# Split the data into training and test sets
set.seed(123)
sample_index <- sample(1:nrow(iris), nrow(iris) * 0.7) # 70% for training
train_data <- iris[sample_index, ]
test_data <- iris[-sample_index, ]

# Train a Support Vector Machine (SVM) classifier
svm_model <- svm(Species ~ ., data = train_data, kernel = "radial")

# Step 8: Make Predictions on the Test Set
svm_predictions <- predict(svm_model, test_data)

# Step 9: Evaluate the Model
confusion_matrix <- table(svm_predictions, test_data$Species)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))
comparison_df <- data.frame(Actual = test_data$Species, Predicted = svm_predictions)
print(comparison_df)