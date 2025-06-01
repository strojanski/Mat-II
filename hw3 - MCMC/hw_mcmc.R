setwd("C:/Users/sebas/one/OneDrive/grive/faks/masters/y1/2nd semester/Mathematics II/Gradient Descent/Mat-II/hw3 - MCMC")
getwd()

fn <- function(x) {
  return(x^(-3/4) * exp(-x))
}

integral <- pgamma(1, shape=0.25, lower.tail=TRUE)
print(integral)

x_vals <- seq(0.001, 1, length.out=1000)
plot(x_vals, fn(x_vals), type="l", col="blue",
     main=expression(f(x) == x^{-3/4} * e^{-x}),
     xlab="x", ylab="f(x)", lwd=2)

y_vals = cumsum(sapply(x_vals, fn))
plot(x_vals, y_vals, type="l", col="blue",
     main=expression(f(x) == x^{-3/4} * e^{-x}),
     xlab="x", ylab="f(x)", lwd=2)
