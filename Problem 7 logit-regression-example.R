# Logistic regression, also called a logit model, is used to model dichotomous outcome variables. 
# In the logit model the log odds of the outcome is modeled as a linear combination of the predictor variables.

library(aod)
library(ggplot2)
library(Rcpp)



mydata <- read.csv(system.file("binary.csv", package="datapack"))

mydata$rank <- factor(mydata$rank)
mylogit <- glm(admit ~ gre + gpa + rank, data = mydata, family = "binomial")
newdata1 <- with(mydata, data.frame(gre = mean(gre), gpa = mean(gpa), rank = factor(1:4)))
newdata1$rankP <- predict(mylogit, newdata = newdata1, type = "response")
newdata2 <- with(mydata, data.frame(gre = rep(seq(from = 200, to = 800, length.out = 100), 4), gpa = mean(gpa), rank = factor(rep(1:4, each = 100))))
newdata3 <- cbind(newdata2, predict(mylogit, newdata = newdata2, type="link", se=TRUE))
newdata3 <- within(newdata3, {
    PredictedProb <- plogis(fit)
    LL <- plogis(fit - (1.96 * se.fit))
    UL <- plogis(fit + (1.96 * se.fit))
})

# Plot the predicted probability for gre
ggplot(newdata3, aes(x = gre, y = PredictedProb)) +
    geom_ribbon(aes(ymin = LL, ymax = UL, fill = rank), alpha = .2) +
    geom_line(aes(colour = rank), size=1)

# Write the plot out
ggsave(sprintf("%s/gre-predicted.png", tempdir()))
