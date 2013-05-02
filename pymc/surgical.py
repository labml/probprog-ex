"""
Example: 'Surgical: Institutional ranking' (simple version) 
From: http://www.openbugs.info/Examples/Surgical.html

The example considers mortality rates in 12 hospitals performing cardiac
surgery in babies.

r[i] represents the number os deaths for hospital i. Each r is modeled as 
a Binomial distribution, with:

$r_i = Binomial(p_i , n_i)

p[i] is modeled with a Beta distribution, so the failure probabilities are
independent for each hospital. So, for a non-informative prior distribution 
to p is:

$p_i = Beta(1.0, 1.0)$

The Binomial distribution gets each hospital mortality as its trial number
and p as the probability of success (in this case, the success is
the death rate)

Original BUGS model:

model
    {
    for( i in 1 : N ) {
        p[i] ~ dbeta(1.0, 1.0)
        r[i] ~ dbin(p[i], n[i])
    }
    }

Data:

list(n = c(47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360),
     r = c(0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24),
     N = 12)

Inits:

list(p = c(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
"""

import numpy as np
import pymc
from pymc import *



n = [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360]
r = [0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24]
p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

#N = 12
N = np.ones(12, dtype=int)

"""
In pymc, instead of looping, just pass an array of values to
the distribution method and the result will be an array

If one variable won't be traced, set observed=True
"""
p_beta = pymc.Beta("beta", alpha=1.0, beta=1.0, value=p)

r_bin = pymc.Binomial("binomial", n=n, p=p_beta, value=r, observed=True)

# Fiting the model with the variables
model = pymc.MCMC([p_beta,r_bin])

# Sampling with 20k iterations
model.sample(20000,5000,2)

# Summary
print model.summary()
