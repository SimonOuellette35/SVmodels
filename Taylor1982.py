import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import csv

# The following shows the difference between Clark (1973) and Taylor (1982)
N = 5000

returns = []
with open('stock_returns.csv', 'rb') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        returns.append(float(row[0]))

returns = returns[:N]

# Clark
def ClarkDGP(mu, sigmaT):
    clark = []
    for _ in range(N):
        time_process = np.random.normal(mu, sigmaT)
        r = np.random.normal(0., np.exp(time_process))
        clark.append(r)

    return np.array(clark)

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

# interval = 100
# with pm.Model() as clark_model:
#     sigmaT = pm.Uniform('sigmaT', lower=0.001, upper=10., testval=0.1)
#     mu = pm.Uniform('mu', lower=-7., upper=-3., testval=-5.3)
#
#     time_process = pm.Normal('time_process', mu=mu, sd=sigmaT, shape=100)
#     time_process = tt.repeat(time_process, N // interval)[:-1]
#
#     pm.Normal('obs', mu=0., sd=pm.math.exp(time_process), observed=returns)
#
#     trace = pm.sample(2000, tune=2000)
#
# clark_sigmaT = np.mean(trace['sigmaT'])
# clark_mu = np.mean(trace['mu'])
#
# print "Clark (1973) model parameters:"
# print "SigmaT = ", clark_sigmaT
# print "Mu = ", clark_mu
clark_sigmaT = 0.57
clark_mu = -4.91

with pm.Model() as taylor_model:
    sigmaT = pm.Uniform('sigmaT', lower=0.001, upper=.2, testval=0.05)
    rhos = pm.Uniform('rhos', lower=-1., upper=1., shape=5)
    mu = pm.Uniform('mu', lower=-7., upper=-3., testval=-5.)

    time_process = pm.AR('time_process', rho=rhos, sd=sigmaT, shape=N)

    pm.Normal('obs', mu=0., sd=pm.math.exp(mu + time_process), observed=returns)

    mean_field = pm.fit(30000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01))
    trace = mean_field.sample(1000)

taylor_sigmaT = np.mean(trace['sigmaT'])
taylor_mu = np.mean(trace['mu'])
taylor_rhos = np.mean(trace['rhos'], axis=0)

print "Taylor (1982) model parameters:"
print "SigmaT = ", taylor_sigmaT
print "Mu = ", taylor_mu
print "Rhos = ", taylor_rhos

print "Generating data for Clark DGP"
clark = ClarkDGP(clark_mu, clark_sigmaT)

print "Generating data for Taylor DGP"
taylor = TaylorDGP(taylor_mu, taylor_sigmaT, taylor_rhos)

print "Plotting..."

fig, axarr = plt.subplots(2, 3)

axarr[0][0].hist(taylor)
axarr[0][0].set_title("Taylor")

axarr[0][1].hist(clark)
axarr[0][1].set_title("Clark")

axarr[1][0].plot(taylor)
axarr[1][0].set_title("Taylor")

axarr[1][1].plot(clark)
axarr[1][1].set_title("Clark")

axarr[0][2].hist(returns)
axarr[0][2].set_title("Real")

axarr[1][2].plot(returns)
axarr[1][2].set_title("Real")

plt.show()
