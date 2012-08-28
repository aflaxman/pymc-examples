# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pymc as mc
import pymc.gp as gp

# <markdowncell>

# 1. Start with a simple GP example:
# ==================================

# <codecell>

x = np.arange(0., 10., 1.)
xx=arange(0,10,.01)

M = gp.Mean(lambda x: 0.*x)
C = gp.Covariance(eval_fun=gp.cov_funs.matern.euclidean,
                  diff_degree=2., amp=1., scale=5.)

f = gp.GPSubmodel('f', M, C, x)

m = mc.MCMC([f])
%time m.sample(iter=10000, burn=5000, thin=1)

# <codecell>

figure(figsize=(8,4))
subplot(1,2,1)
for k in [0, 1000, 2000, 3000, 4000]:
    plot(x, f.f_eval.trace()[k,:], 'ks:')
    plot(xx,f.f.trace()[k](xx), 'k-')
ylabel('f(x)',rotation='horizontal')
title('5 MCMC draws')
grid()

subplot(1,2,2)
acorr(f.f_eval.trace()[:,5], detrend=mlab.detrend_mean, maxlags=300)
title('acorr for f(5)')

# <markdowncell>

# 2. Now add a bound on the (approximate) derivative at a point
# =============================================================

# <codecell>

x = np.arange(0., 10., 1.)
xx=arange(0,10,.01)

M = gp.Mean(lambda x: 0.*x)
C = gp.Covariance(eval_fun=gp.cov_funs.matern.euclidean,
                  diff_degree=2., amp=1., scale=5.)

f = gp.GPSubmodel('f', M, C, x)

@mc.potential
def deriv_bound(f=f, x_0=5., c=0., eps=.2):
    # use secant approximation of derivative
    Df = (f.f(x_0+eps) - f.f(x_0-eps)) / (2*eps)
    
    # "soft" constraint may help convergence
    return -1000. * (Df - c)**2.

m = mc.MCMC([f, deriv_bound])
%time m.sample(iter=10000, burn=5000, thin=1)

# <codecell>

figure(figsize=(8,4))
subplot(1,2,1)
for k in [0, 1000, 2000, 3000, 4000]:
    plot(x, f.f_eval.trace()[k,:], 'ks:')
    plot(xx,f.f.trace()[k](xx), 'k-')
ylabel('f(x)',rotation='horizontal')
title('5 MCMC draws')
grid()

subplot(1,2,2)
acorr(f.f_eval.trace()[:,5], detrend=mlab.detrend_mean, maxlags=300)
title('acorr for f(5)')

# <markdowncell>

# Note that autocorrelation function shows nonzero correlation out to 200 or more lags, meaning convergence is 200 times slower when this constraint is included.

# <markdowncell>

# 3. This approach can also add a bound on the approx derivative for many points
# ==============================================================================

# <codecell>

x = np.arange(0., 10., 1.)
xx=arange(0,10,.01)

M = gp.Mean(lambda x: 0.*x)
C = gp.Covariance(eval_fun=gp.cov_funs.matern.euclidean,
                  diff_degree=2., amp=1., scale=5.)

f = gp.GPSubmodel('f', M, C, x)
Df = mc.Lambda('Df', lambda f=f, x=x: np.diff(f.f_eval) / np.diff(x))    # use secant approximation of derivative

@mc.potential
def deriv_bound(Df=Df, ub=0.):
    return -1000. * np.sum(np.maximum(0, Df-ub)**2)

m = mc.MCMC([f, deriv_bound])
%time m.sample(iter=10000, burn=5000, thin=1)

# <markdowncell>

# Note that MCMC is much faster if you don't evaluate GP off of f_eval mesh.

# <codecell>

figure(figsize=(8,4))
subplot(1,2,1)
for k in [0, 1000, 2000, 3000, 4000]:
    plot(x, f.f_eval.trace()[k,:], 'ks:')
    plot(xx,f.f.trace()[k](xx), 'k-')
ylabel('f(x)',rotation='horizontal')
title('5 MCMC draws')
grid()

subplot(1,2,2)
acorr(f.f_eval.trace()[:,5], detrend=mlab.detrend_mean, maxlags=300)
title('acorr for f(5)')

# <markdowncell>

# Extensions you may consider
# ===========================
# 
# 1. Speed up example 2, by choosing the mesh for f_eval to respect the $\epsilon$ value in the derivative approximation.
# 2. Fix example 3, so that it is decreasing for $x>9$.
# 3. Experiment with different scale parameters in the Matern covariance function.  I think this can break the derivative constraint approximations.  Can you fix it?
# 4. (Advanced) Experiment with alternative step methods for the MCMC. Can a custom step method achieve convergence speeds more like those of the GPStepMethod has for the unconstrained case?

