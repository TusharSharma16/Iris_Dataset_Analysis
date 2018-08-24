#Importing the dataset
dataset = read.csv('Iris.csv')

#Getting the dimension of the dataset
dim(dataset)

#Summarizing the details of iris dataset
summary(dataset)

# histogram functions
hist(dataset$SepalLengthCm, col = "blue", xlab = "Sepal Length", main = 
       "Histogram of Sepal Length of Iris Data")
hist(dataset$SepalWidthCm, col = "blue", xlab = "Sepal Width", main = 
       "Histogram of Sepal Width of Iris Data")
hist(dataset$PetalLengthCm, col = "blue", xlab = "Petal Length", main = 
       "Histogram of Petal Length of Iris Data")
hist(dataset$PetalWidthCm, col = "blue", xlab = "Petal Width", main = 
       "Histogram of Petal Width of Iris Data")

#Creating piechart
table(dataset$Species)
pie(table(dataset$Species), main = "Pie Chart of the Iris data set Species", 
    col = c("orange1", "chocolate", "coral"), radius = 1)

#Covariance between Parameters
cov(dataset[, 1:4])

#Coorelation between parameters
cor(dataset[, 1:4])

#Data Distribution using box plot
par(mfrow=c(2,2))
boxplot(SepalLengthCm  ~ Species, dataset, main = "Sepal Length wrt Species", col = "lightpink3")
boxplot(SepalWidthCm   ~ Species, dataset, main = "Sepal Width wrt Species", col = "antiquewhite1")
boxplot(PetalLengthCm  ~ Species, dataset, main = "Petal Length wrt Species", col = "lightskyblue4")
boxplot(PetalWidthCm  ~ Species, dataset, main = "Petal Width wrt Species", col = "orange1")

#SepalLength Vs SepalWidth distribution using ggplot
library(ggplot2)
g <- ggplot(dataset, aes(x = SepalLengthCm, y = SepalWidthCm))
g <- g + geom_point(aes(shape = factor(dataset$Species), colour = factor(dataset$Species)))
g <- g + ggtitle (" SepalLength Vs SepalWidth wrt Species" )
g <- g + stat_smooth(method= lm)
g
