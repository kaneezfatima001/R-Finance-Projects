##### TESLA #####
## tesla weekly stocks prices from 25 jan 2017 to 24 aug 2022
## and we are using high prices since others are somehow inadequate
library(fBasics); library(forecast); library(fGarch); library(ggplot2)
library(lmtest)

#### PART 1 ####
tsla = read.csv("TSLA.csv")
dim(tsla)
summary(tsla$High)
par(mfrow = c(1, 1))
year = c(1:293) / 52 + 2017
plot(year, tsla$High, type = "l", ylab = "Weekly High stock price", 
     main = "TSLA")

yt = diff(log(tsla$High))
length(yt)
plot(year[-1], yt, type = "l", xlab = "Year", ylab = "Weekly log return",
     main = "TSLA")

# acf and pacf
ggtsdisplay(yt, xlab = "Time", ylab = "Weekly log return",
            main = "TSLA")

# Basic Statistics
basicStats(yt)

# mean test
t.test(yt) # reject H0, mean != 0

# skewness test
t1 = skewness(yt) / sqrt(6/length(yt)); t1
pv1 = 2 * pnorm(t1, lower.tail = FALSE); pv1  # reject H0

# kurtosis test
t2 = kurtosis(yt) / sqrt(24/length(yt)); t2
pv2 = 2 * pnorm(t2, lower.tail = FALSE); pv2  # reject H0

# JB test to verify
normalTest(yt, "jb") # not normal

# density plot
plot(density(yt)$x, density(yt)$y, type = "l", xlab = "Weekly log returns", 
     ylab = "density", main = "TSLA")
x = seq(min(yt), max(yt), 0.001)
y = dnorm(x, mean(yt), sd(yt))
lines(x, y, type = "l", lty = 2, col = "blue", cex = 2)


#### PART 2 ####
# order identification
# ACF and PACF cuts off equally => MA(1)
ar.mle(yt)$order # ARMA(1, 0)
auto.arima(yt) # ARMA(0, 2)


m1 = arima(yt, order = c(1, 0, 0)); coeftest(m1)
m1 = arima(yt, order = c(1, 0, 0), include.mean = F); coeftest(m1) # since the mean is insignificant
Box.test(m1$residuals, 10, "Ljung")
Box.test(m1$residuals^2, 10, "Ljung")

m2 = arima(yt, order = c(0, 0, 2)); coeftest(m2)
m3 = arima(yt, order = c(0, 0, 1)); coeftest(m3)
Box.test(m3$residuals, 10, "Ljung")
Box.test(m3$residuals^2, 10, "Ljung")

# plotting of the residuals of ARMA model
par(mfrow = c(1, 1))
plot(year[-1], m1$residuals, type = "l", xlab = "Year",
     main = "Residuals")
par(mfrow = c(1, 2))
acf(m1$residuals)
acf(m1$residuals^2)

# both also ok
# choose ar1 for tentative identification
# to test with ma1 later on to find the best fit model

Box.test(yt, 10, "Ljung")
Box.test(abs(yt), 10, "Ljung")

par(mfrow = c(1, 2))
acf(yt)
acf(abs(yt))

t.test(yt) # mean != 0

xt = yt - mean(yt)

Box.test(xt^2, 10, "Ljung")

par(mfrow = c(1, 1))
plot(year[-1], xt^2, type = "l", xlab = "Year", ylab = " ",
     main = expression(epsilon[t]^2))
par(mfrow = c(1, 2))
acf(xt^2, lag = 36, main = expression(epsilon[t]^2), ylim = c(-0.2, 1))
pacf(xt^2, lag = 36, main = expression(epsilon[t]^2), ylim = c(-0.2, 1))
par(mfrow = c(1, 1))

# from the acf and pacf of xt^2 => GARCH(1, 1) or GARCH(4, 0) or GARCH(6, 0)
auto.arima(xt^2) # GARCH(0, 2)
ar.mle(xt^2)$order # GARCH(11, 0)

hm1 = garchFit(~ arma(1, 0) + garch(11, 0), data = yt, trace = F); summary(hm1)

hm2 = garchFit(~ arma(1, 0) + garch(1, 0), data = yt, trace = F); summary(hm2)

hm3 = garchFit(~ arma(1, 0) + garch(1, 1), data = yt, trace = F); summary(hm3)

hm4 = garchFit(~ arma(0, 1) + garch(1, 1), data = yt, trace = F); summary(hm4)

# hm3 with ar(1) aic = -2.481023
# hm4 with ma(1) aic = -2.484192
# so apply ma(1)
# mean is insignificant

hm4 = garchFit(~ arma(0, 1) + garch(1, 1), data = yt, trace = F,
               include.mean = F); summary(hm4)

hm5 = garchFit(~ arma(0, 1) + garch(1, 1), data = yt, trace = F,
               cond.dist = "std", include.mean = F); summary(hm5)

hm6 = garchFit(~ arma(0, 1) + garch(1, 1), data = yt, trace = F,
               cond.dist = "sstd", include.mean = F); summary(hm6)

# skewness test
(1.0064198 - 1) / 0.0814684
# DNR H0, not skewed => hm5 is preferred

# alpha0 always +ve due to parameter restrictions
# what if we use ar(1) => just to confirm
hm5.test = garchFit(~ arma(1, 0) + garch(1, 1), data = yt, trace = F,
               cond.dist = "std", include.mean = F); summary(hm5.test)
# hm5 with ar(1) aic = -2.561426
# hm5 with ma(1) aic = -2.562435
# hm5 with ma(1) is preferred

# plotting
par(mfrow = c(1, 3))
plot(hm5)
10
11
13
0

par(mfrow = c(1, 1))


# since the mean is insignificant
mu = 0
sigma = volatility(hm5)

cv = qstd(0.975, nu = 5.2458354); cv

upp.lmt = mu + cv * sigma
low.lmt = mu - cv * sigma

# brief predict on 5-step ahead
predict(hm5, n.ahead = 5, plot = T)

# to answer the assignment question
predict(hm5, n.ahead = 1, plot = T)

# plotting the interval on the time series
plot(year[-1], yt, type = "l", xlab = "Year",
     ylab = "Weekly Log Return", main = "TSLA")

lines(year[-1], upp.lmt, lty = 2, lwd = 1.5, col = "blue")
lines(year[-1], low.lmt, lty = 2, lwd = 1.5, col = "blue")
abline(h = 0, col = "red", lwd = 2)
