"""
Example: 'Air: Berkson measurement error'
From: [1] http://www.openbugs.info/Examples/Air.html
      [2] http://www.mrc-bsu.cam.ac.uk/bugs/thebugsbook/examples/html/
          Chapter-9-Issues/Example-9_3_3-airpollution.html 
      
The example use an approximate maximum likelihood approach to analyse the 
data on reported respiratory illness versus exposure to nitrogen dioxide
(NO 2 ) in 103 children. Details about the model are in [1].

The model used in this port is described in [2].

- BUGS model:

model {
for (j in 1:3) {
    y[j] ~ dbin(p[j], n[j])
    logit(p[j]) <- theta[1] + theta[2]*z[j]
    z[j] ~ dnorm(mu[j], 0.01232)
    mu[j] <- alpha + beta*x[j]
}
theta[1] ~ dnorm(0, 0.0001)
theta[2] ~ dnorm(0, 0.0001)
}

- Data:

list(y = c(21, 20, 15), 
     n = c(48, 34, 21), 
     x = c(10, 30, 50), 
     alpha = 4.48, 
     beta = 0.76)

- Inits:

list(theta = c(0.0, 0.0),
     z = c(0.0, 0.0, 0.0))

- The expected results are:

       node   mean      sd       MC error   median    start   sample
   theta[1]   -0.8096   0.8559   0.03736    -0.6557   12001   20000
   theta[2]   0.04207   0.03144  0.001313   0.03667   12001   20000
       z[1]   12.8      8.299    0.2011     12.98     12001   20000
       z[2]   27.43     7.474    0.08438    27.37     12001   20000
       z[3]   41.43     8.56     0.1437     41.23     12001   20000    
"""
import numpy as np
import pymc
from pymc import *

alpha = 4.48        
beta = 0.76         

z = [0.0, 0.0, 0.0]
y = [21, 20, 15]
n = [48, 34, 21]
x = [10, 30, 50]


theta1 = pymc.Normal("theta1", mu=0, tau=0.0001)
theta2 = pymc.Normal("theta2", mu=0, tau=0.0001)

"""
The loop from BUGS model isn't necessary. Just pass an arry of values
and the result will be an array

To translate this following BUGS code
    y[j] ~ dbin(p[j], n[j])
    logit(p[j]) <- theta[1] + theta[2]*z[j]

to pymc is needed to calculate and assign values for the array z 
and later do the logit of each value.

The p_logit gets the right value for z[] and p=invlogit(p_logit)
pass the invlog og each z[] (to avoid negative numbers) to the p parameter
in Binomial()
"""
p_logit = [theta1.value + theta2.value * z_j for z_j in z]
y_bin = pymc.Binomial("y_bin", n=n, p=pymc.invlogit(p_logit), value=y,
                     observed=True)

# mu_norm is calculated similarly to p_logit
mu_norm = [alpha + beta * x_j for x_j in x]
z_norm = pymc.Normal("z_norm", mu=mu_norm, tau=0.01232, value=z)

# Modeling (only thetas and z values are described
model = pymc.MCMC([theta1, theta2, y_bin, z_norm])

# Sampling with 20k itetarions
model.sample(20000,5000,2)

model.summary()
