# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <markdowncell>

# PyMC Pandas Example
# ===================
# 
# This example project shows how to fit a fixed effects Poisson model with PyMC.  It uses pandas Series and DataFrame objects to store data in a classy way.

# <codecell>

import pylab as pl
import pymc as mc
import pandas

# <markdowncell>

# 1. Simulate Noisy Data
# ----------------------

# <codecell>

# simulate data with known distribution

N = 100
X = pandas.DataFrame({'constant': pl.ones(N), 'cov_1': pl.randn(N)})

beta_true = pandas.Series(dict(constant=100., cov_1=20.))
mu_true = pl.dot(X, beta_true)

Y = mc.rpoisson(mu_true)

# <codecell>

# explore the data a little bit graphically

pl.figure(figsize=(11,4.25))

pl.subplot(1,2,1)
pl.hist(Y)
pl.xlabel('Observed Count')

pl.subplot(1,2,2)
pl.plot(X['cov_1'], Y, '.')
pl.xlabel('Covariate 1')
pl.ylabel('Observed Count')

# <markdowncell>

# 2. Model data with PyMC
# -----------------------
# 
# The following code creates a fixed effect Poisson model 
# where the observed data stored in Y is explained by the
# covariate data in X, according to the formula:
# 
# $$
# Y_i \sim \text{Poisson}(\mu_i),
# $$
# $$
# \mu_i = X_i\cdot \beta.
# $$

# <codecell>

# the simplest approach doesn't work with PyMC 2.1alpha, but it does with 2.2grad
print 'pymc version:', mc.__version__

beta = mc.Uninformative('beta', value=[Y.mean(), 0.])
mu_pred = mc.Lambda('mu_pred', lambda beta=beta, X=X: pl.dot(X, beta))
Y_obs = mc.Poisson('Y_obs', mu=mu_pred, value=Y, observed=True)

# <codecell>

m = mc.Model([beta, mu_pred, Y_obs])
%time mc.MCMC(m).sample(10000, 5000, 5, progress_bar=False)

# <codecell>

mc.Matplot.plot(beta, common_scale=False)
print '\ntrue value of beta\n', beta_true
print '\npredicted:'
print pandas.DataFrame({'mean':beta.stats()['mean'],
                        'lb':beta.stats()['95% HPD interval'][:,0],
                        'ub':beta.stats()['95% HPD interval'][:,1]},
                       columns=['mean','lb','ub'])

# <markdowncell>

# 2a. TODO: Integrate PyMC and Pandas further
# ---------------------------------------------

# <codecell>

# making beta.value a pandas.Series would be slightly cooler than the above

@mc.stochastic
def beta(value=pandas.Series(dict(constant=Y.mean(), cov_1=0))):
    return 0.
mu_pred = mc.Lambda('mu_pred', lambda beta=beta, X=X: pl.dot(X, beta))
Y_obs = mc.Poisson('Y_obs', mu=mu_pred, value=Y, observed=True)

# <codecell>

beta.value

# <codecell>

# unfortunately the pandas.Series becomes a numpy.array during MCMC
m = mc.Model([beta, mu_pred, Y_obs])
mc.MCMC(m).sample(10000, 5000, 5, progress_bar=False)

# <codecell>

beta.value # in a pandas-centric version of PyMC, this would still be a pandas.Series

# <codecell>


