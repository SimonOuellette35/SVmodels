import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import csv

# The following shows the difference between Wiggins (1987) and Taylor (1982)

N = 10000

returns = []
with open('stock_returns.csv', 'rb') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        returns.append(float(row[0]))

returns = returns[:N]

# Taylor
# with pm.Model() as taylor_model:
#     sigmaT = pm.Uniform('sigmaT', lower=0.001, upper=.2, testval=0.05)
#     rhos = pm.Uniform('rhos', lower=-1., upper=1., shape=5)
#     mu = pm.Uniform('mu', lower=-7., upper=-3., testval=-5.)
#
#     time_process = pm.AR('time_process', rho=rhos, sd=sigmaT, shape=N)
#
#     pm.Normal('obs', mu=0., sd=pm.math.exp(mu + time_process), observed=returns)
#
#     mean_field = pm.fit(20000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01))
#     trace = mean_field.sample(1000)
#
# taylor_sigmaT = np.mean(trace['sigmaT'])
# taylor_mu = np.mean(trace['mu'])
# taylor_rhos = np.mean(trace['rhos'], axis=0)

taylor_sigmaT = 0.2
taylor_mu = -4.53
taylor_rhos = [0.30188776, 0.25813007, 0.20395745, 0.13450001, 0.06772271]

print "Taylor (1982) model parameters:"
print "SigmaT = ", taylor_sigmaT
print "Mu = ", taylor_mu
print "Rhos = ", taylor_rhos

# Wiggins
def WigginsDGP(volatility_mu, volatility_theta, volatility_sigma):

    wiggins = []
    volatility = [volatility_mu]
    for _ in range(N):
        v = volatility[-1] + volatility_theta * (volatility_mu - volatility[-1]) \
            + np.random.normal(0., volatility_sigma)

        volatility.append(v)
        r = np.random.normal(0., np.exp(v))

        wiggins.append(r)

    return np.array(wiggins)

# Taylor
def TaylorDGP(mu, sigmaT, rhos):
    taylor = []
    time_process = [0.] * 5
    for _ in range(N):
        v = rhos[0] * time_process[-1] \
            + rhos[1] * time_process[-2] \
            + rhos[2] * time_process[-3] \
            + rhos[3] * time_process[-4] \
            + rhos[4] * time_process[-5] \
            + np.random.normal(0., sigmaT)

        time_process.append(v)
        r = np.random.normal(0., np.exp(mu + v))

        taylor.append(r)

    return np.array(taylor)

import pymc3.distributions.timeseries as ts
with pm.Model() as wiggins_model:

    volatility_theta = pm.Uniform('volatility_theta', lower=0., upper=1., testval=0.5)
    volatility_mu = pm.Normal('volatility_mu', mu=-5., sd=.1, testval=-5)
    volatility_sigma = pm.Uniform('volatility_sigma', lower=0.001, upper=.2, testval=0.05)

    sde = lambda x, theta, mu, sigma: (theta * (mu - x), sigma)
    volatility = ts.EulerMaruyama('volatility',
                                  1.0,
                                  sde,
                                  [volatility_theta, volatility_mu, volatility_sigma],
                                  shape=len(returns),
                                  testval=np.ones_like(returns))

    pm.Normal('obs', mu=0., sd=pm.math.exp(volatility), observed=returns)

    mean_field = pm.fit(30000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01))
    trace = mean_field.sample(2000)

print "Wiggins (1987) model parameters:"

volatility_mu = np.mean(trace['volatility_mu'])
volatility_theta = np.mean(trace['volatility_theta'])
volatility_sigma = np.mean(trace['volatility_sigma'])
print "Volatility mu: ", volatility_mu
print "Volatility theta: ", volatility_theta
print "Volatility sigma: ", volatility_sigma

print "Generating data for Wiggins DGP"
wiggins = WigginsDGP(volatility_mu, volatility_theta, volatility_sigma)

print "Generating data for Taylor DGP"
taylor = TaylorDGP(taylor_mu, taylor_sigmaT, taylor_rhos)

print "Plotting..."

fig, axarr = plt.subplots(2, 3)

axarr[0][0].hist(taylor)
axarr[0][0].set_title("Taylor")

axarr[0][1].hist(wiggins)
axarr[0][1].set_title("Wiggins")

axarr[1][0].plot(taylor)
axarr[1][0].set_title("Taylor")
axarr[1][0].set_ylim(bottom=-0.15, top=0.15)

axarr[1][1].plot(wiggins)
axarr[1][1].set_title("Wiggins")
axarr[1][1].set_ylim(bottom=-0.15, top=0.15)

axarr[0][2].hist(returns)
axarr[0][2].set_title("Real")

axarr[1][2].plot(returns)
axarr[1][2].set_title("Real")
axarr[1][2].set_ylim(bottom=-0.15, top=0.15)

plt.show()
