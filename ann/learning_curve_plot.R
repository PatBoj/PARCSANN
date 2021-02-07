library(ggplot2)
library(gganimate)
library(transformr)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

errors <- data.frame(read.csv("error.csv", header = FALSE, sep = ","))
errors$average <- rowMeans(errors[, -1]) * 100

avg <- ggplot(errors, aes(x = V1, y = average)) +
  geom_point(size = 1.5) +
  xlab("epoch (iteration)") +
  ylab("accuracy (mean absolute error) in percenteage") +
  theme(
    text = element_text(size = 20),
    axis.text = element_text(color = "black"),
    panel.border = element_rect(fill = NA),
    panel.background = element_blank(),
    legend.key = element_rect(fill = NA, color = NA),
    legend.background = element_rect(fill = (alpha("white", 0))),
    legend.title = element_blank(),
    legend.box.background = element_rect(colour = "black")
  )

png("avg_error.png", width = 800, height = 500)
print(avg)
dev.off()

z1 <- rnorm(10,100,15)
z2 <- rnorm(25,100,15)
z3 <- rnorm(100,100,15)
z4 <- rnorm(500,100,15)
z5 <- rnorm(2000,100,15)

z1 <- as.matrix(z1, nrows=10, ncols=1)
z2 <- as.matrix(z2, nrows=25, ncols=1)
z3 <- as.matrix(z3, nrows=100, ncols=1)
z4 <- as.matrix(z4, nrows=500, ncols=1)
z5 <- as.matrix(z5, nrows=2000, ncols=1)

z1 <- as.data.frame(z1)
z2 <- as.data.frame(z2)
z3 <- as.data.frame(z3)
z4 <- as.data.frame(z4)
z5 <- as.data.frame(z5)

colnames(z1) <- "IQ"
colnames(z2) <- "IQ"
colnames(z3) <- "IQ"
colnames(z4) <- "IQ"
colnames(z5) <- "IQ"

dataset <- c(rep(1,10),rep(2,25),rep(3,100),rep(4,500),rep(5,2000))

fullmat <- matrix(0,2635,2) 
fullmat[,1] <- dataset
partmat <- rbind(z1,z2,z3,z4,z5)
fullmat[,2] <- partmat[,1]
colnames(fullmat) <- c("dataset","IQ")
fullmat <- as.data.frame(fullmat)

anim_plot1 <- ggplot(errors[,-ncol(errors)], aes(V1)) + 
  geom_histogram(col = "black",fill = "blue") + 
  transition_states(errors[,1],2,4) + 
  view_follow(fixed_x = TRUE)

anim_plot1