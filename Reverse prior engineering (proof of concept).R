#Notation key

#mu_ystar: backward pp mean
#mu_h_given_x: prior mean
#mu_ystar_given_y: forward pp mean

#sigma2: likelihood variance
#sigma2_ystar: backward pp variance
#sigma2_h_given_x: prior variance
#sigma2_ystar_given_y: forward pp variance

#Data generation
set.seed(42)
true_fn <- function(x) x * sin(exp(x / 35))
n <- 1000
sigma2 <- 1
x_vals <- 100-rgamma(n,1.5,0.05)
y_vals <- true_fn(x_vals) + rnorm(n, mean = 0, sd = sqrt(sigma2))
df <- data.frame(x = x_vals, y = y_vals)
n <- nrow(df)
tol <- 0.005

#Divide data into training and testing
train_idx <- sample(seq_len(n), size=floor(0.8*n), replace=FALSE)
df_train <- df[train_idx,]
df_test <- df[-train_idx,]

#Determine pp mean and var
mu_ystar_finder <- lm(y~splines::bs(x,df=50),data=df_train)
mu_ystar <- function(input) predict(mu_ystar_finder,newdata=data.frame(x=input))
sigma2_ystar_finder <- function(sig2,data,penalty=0.01,level=0.95){
  (mean(qnorm((1-level)/2,predict(mu_ystar_finder,newdata=data.frame(x=data$x)),sqrt(sig2)) < data$y &
          qnorm((1+level)/2,predict(mu_ystar_finder,newdata=data.frame(x=data$x)),sqrt(sig2)) > data$y) - 
     level)^2 + penalty*log(sig2)
}
about <- function(input,x,tolerance) { #input=scalar,x=vector,tolerance=scalar
  which(abs((input-x)/input) < tolerance)
}
ems <- vector("numeric")
for(i in seq_along(df_train$x)) {
  y_i <- df_train[about(df_train$x[i],df_train$x,tol),"y"]
  while(length(y_i) == 0) {
    tol <- tol*1.5
    y_i <- df_train[about(input,df_train$x,tol),"y"]
  }
  ems[i] <- length(y_i)
}
em <- max(ems) #Set em at worst-case scenario so that every prior variance is adequately bounded
sigma2_ystar <- optim(par=10,sigma2_ystar_finder,data=df_train,method="Brent",
                  lower=sigma2,upper=(em+1)/em*sigma2)$par
if(sigma2_ystar > 2*sigma2) cat("sigma2_ystar too big: ",sigma2_ystar," > ",2*sigma2)

#Determine prior mean and var
mu_h_given_x <- function(input,y) {
  m <- length(y)
  (sum(y)*sigma2_ystar-sum(y)*sigma2-mu_ystar(input)*sigma2)/(sigma2_ystar*m-sigma2*(m+1))
}
sigma2_h_given_x <- function(y) {
  m <- length(y)
  (sigma2_ystar-sigma2)/(1-m*(sigma2_ystar/sigma2-1))
}

#Prepare to assess UQ on test data, computing pp on ys where the xs approximately coincide
mu_ystar_given_y <- function(input,training,tolerance,echo=T) {
  y_i <- training[about(input,training$x,tolerance),"y"]
  while(length(y_i) == 0) {
    tolerance <- tolerance*1.5
    y_i <- training[about(input,training$x,tolerance),"y"]
    }
  m <- length(y_i)
  if((m+1)/m <= sigma2_ystar/sigma2) {
    stop(paste0("Error: mu_ystar not estimated precisely enough. Decrease m=",
                m," to below ",round(1/(sigma2_ystar/sigma2-1),3),
                ",\n or estimate the mean model better so that sigma2_y*/sigma2_l < ",
                round((m+1)/m,3),". Currently sigma2_y*=",round(sigma2_ystar,3),"."))
  }
  out <- list((sum(y_i)*sigma2_h_given_x(y_i) + mu_h_given_x(input,y_i)*sigma2)/(m*sigma2_h_given_x(y_i)+sigma2),
              m,tolerance,y_i)
  if(echo) cat("Forward pp mean estimated as",round(out[[1]],3),"for this input.")
  invisible(out)
}
sigma2_ystar_given_y <- function(input,training,tolerance,echo=T) {
  y_i <- training[about(input,training$x,tolerance),"y"]
  while(length(y_i) == 0) {
    tolerance <- tolerance*1.5
    y_i <- training[about(input,training$x,tolerance),"y"]
  }
  out <- sigma2_h_given_x(y_i)*sigma2/(sigma2_h_given_x(y_i)*length(y_i)+sigma2) + sigma2
  if(echo) cat("Forward pp sigma2 estimated as",round(out,3),"for this input.")
  invisible(out)
}

#Now assess for a single test data point.
tol <- 0.005
test_point <- df_test[1,]
curve(dnorm(x,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
      from=mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]]-3*sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F)),
      to=mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]]+3*sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F)),
      ylab="Forward PP",xlab="y*",col="orange")
polygon(c(seq(qnorm(0.025,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
              qnorm(0.975,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),,1e2),
          qnorm(0.975,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
          qnorm(0.025,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F)))),
        c(dnorm(seq(qnorm(0.025,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
                    qnorm(0.975,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),,1e2),
                mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),0,0),
        col="orange",border="orange")
abline(v=test_point$y,col="purple",lty="dashed",lwd=2)
#Hit

#Do another point.
test_point <- df_test[2,]
curve(dnorm(x,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
      from=mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]]-3*sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F)),
      to=mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]]+3*sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F)),
      ylab="Forward PP",xlab="y*",col="orange")
polygon(c(seq(qnorm(0.025,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
              qnorm(0.975,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),,1e2),
          qnorm(0.975,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
          qnorm(0.025,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F)))),
        c(dnorm(seq(qnorm(0.025,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
                    qnorm(0.975,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),,1e2),
                mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),0,0),
        col="orange",border="orange")
abline(v=test_point$y,col="purple",lty="dashed",lwd=2)
#Hit

#And another one.
test_point <- df_test[3,]
curve(dnorm(x,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
      from=mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]]-3*sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F)),
      to=mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]]+3*sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F)),
      ylab="Forward PP",xlab="y*",col="orange")
polygon(c(seq(qnorm(0.025,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
              qnorm(0.975,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),,1e2),
          qnorm(0.975,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
          qnorm(0.025,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F)))),
        c(dnorm(seq(qnorm(0.025,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),
                    qnorm(0.975,mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),,1e2),
                mu_ystar_given_y(test_point$x,df_train,tol,echo=F)[[1]][[1]],sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))),0,0),
        col="orange",border="orange")
abline(v=test_point$y,col="purple",lty="dashed",lwd=2)
dev.off()

#Loop over all the test points and keep track of the hit rate
hit_rate <- 0
p <- nrow(df_test)
outs <- list(m=vector("numeric"),tol=vector("numeric"),
             mu_ystar_given_y=vector("numeric"),sigma2_h_given_x=vector("numeric"))
for(i in 1:p) {
  test_point <- df_test[i,]
  mu <- mu_ystar_given_y(test_point$x,df_train,tol,echo=F)
  sigma <- sqrt(sigma2_ystar_given_y(test_point$x,df_train,tol,echo=F))
  curve(dnorm(x,mu[[1]],sigma),
        from=mu[[1]]-3*sigma,
        to=mu[[1]]+3*sigma,
        ylab="Forward PP",xlab="y*",col="orange")
  polygon(c(seq(qnorm(0.025,mu[[1]],sigma),
                qnorm(0.975,mu[[1]],sigma),,1e2),
            qnorm(0.975,mu[[1]],sigma),
            qnorm(0.025,mu[[1]],sigma)),
          c(dnorm(seq(qnorm(0.025,mu[[1]],sigma),
                      qnorm(0.975,mu[[1]],sigma),,1e2),
                  mu[[1]],sigma),0,0),
          col="orange",border="orange")
  abline(v=test_point$y,col="purple",lty="dashed",lwd=2)
  if (test_point$y < qnorm(0.975,mu[[1]],sigma) &
    qnorm(0.025,mu[[1]],sigma) < test_point$y) {
      hit_rate <- hit_rate + 1
  }
  #Report m, tol, sigma2_ystar
  outs[["m"]][i] <- mu[[2]]
  outs[["tol"]][i] <- mu[[3]]
  outs[["mu_ystar_given_y"]][i] <- mu[[1]]
  outs[["sigma2_h_given_x"]][i] <- sigma2_h_given_x(mu[[4]])
}
cat("The empirical coverage rate on the test data is",hit_rate/ceiling(n*0.2),"and the nominal rate is 0.95")

plot(df_train$x,df_train$y,cex=0.5,pch=16,xlab="x",ylab="y")
points(df_test$x,df_test$y,cex=0.5,pch=17,col="red3")
points(df_test$x,outs[["mu_ystar_given_y"]],cex=0.5,col="green")
