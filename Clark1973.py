import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import csv
import theano.tensor as tt

# The following shows the difference between constant volatility Gaussian models and Clark (1973)

N = 20000

# Clark
def ClarkDGP(mu, sigmaT):
    clark = []
    for _ in range(N):
        time_process = np.random.normal(mu, sigmaT)
        r = np.random.normal(0., np.exp(time_process))
        clark.append(r)

    return clark

returns = []
with open('stock_returns.csv', 'rb') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        returns.append(float(row[0]))

with pm.Model() as gaussian_model:
    sigma = pm.Uniform('sigma', lower=-7., upper=-3, testval=-5.3)
    pm.Normal('obs', mu=0., sd=pm.math.exp(sigma), observed=returns)

    trace = pm.sample(2000, tune=2000)

sigma = np.mean(trace['sigma'])

print "Gaussian model parameter:"
print "Sigma = ", sigma

interval = 100
with pm.Model() as clark_model:
    sigmaT = pm.Uniform('sigmaT', lower=0.001, upper=10., testval=0.1)
    mu = pm.Uniform('mu', lower=-7., upper=-3., testval=-5.3)

    time_process = pm.Normal('time_process', mu=mu, sd=sigmaT, shape=100)
    time_process = tt.repeat(time_process, N // interval)[:-1]

    pm.Normal('obs', mu=0., sd=pm.math.exp(time_process), observed=returns)

    trace = pm.sample(2000, tune=2000)

sigmaT = np.mean(trace['sigmaT'])
mu = np.mean(trace['mu'])

print "Clark (1973) model parameters:"
print "SigmaT = ", sigmaT
print "Mu = ", mu

print "Generating Clark data..."
clark = ClarkDGP(mu, sigmaT)
print "Generating Gaussian data..."
gaussian = np.random.normal(0., np.exp(sigma), N)

print "Plotting..."

fig, axarr = plt.subplots(2, 3)

axarr[0][0].hist(gaussian)
axarr[0][0].set_title("Gaussian")

axarr[0][1].hist(clark)
axarr[0][1].set_title("Clark")

axarr[0][2].hist(returns)
axarr[0][2].set_title("Real")

axarr[1][0].plot(gaussian)
axarr[1][0].set_title("Gaussian")

axarr[1][1].plot(clark)
axarr[1][1].set_title("Clark")

axarr[1][2].plot(returns)
axarr[1][2].set_title("Real")

plt.show()
