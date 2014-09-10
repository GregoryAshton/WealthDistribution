import numpy as np
import matplotlib.pyplot as plt

N = 10
runN = 100
epsilon = 1e-1
initial_distribution = np.zeros(N) + 1./N
steps = range(runN+1)

def FairShare(distribution, fraction_stolen=1e-2):
    """ Redistribute `distribution' according to a fairshare principle """
    for i in range(len(distribution)):
        j = np.random.randint(0, N)
        #fraction_stolen = 1e-1*np.random.normal(last_distribution[i], 0.1)
        amount_stolen = fraction_stolen*distribution[j]
        distribution[i] += amount_stolen 
        distribution[j] -= amount_stolen
    return distribution

def ProtectTheRich(distribution, fraction_stolen=1e-2):
    """ Redistribute `distribution' with less stolen from the rich """
    for i in range(len(distribution)):
        j = np.random.randint(0, N)
        fraction_stolen = 1e-1*np.random.normal(distribution[i], 0.1)
        amount_stolen = fraction_stolen*distribution[j]
        distribution[i] += amount_stolen 
        distribution[j] -= amount_stolen
    return distribution

distribution = np.expand_dims(initial_distribution, 0)
for i in range(runN):
    #new_distribution = FairShare(distribution[-1])
    new_distribution = ProtectTheRich(distribution[-1])
    distribution = np.append(distribution, [new_distribution], axis=0)

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

average = np.average(distribution, axis=1)
std = np.std(distribution, axis=1)
ax1.errorbar(steps, average, yerr=std)

j_richest = np.argmax(np.average(distribution, axis=0))
richest = distribution[:, j_richest]

j_poorest = np.argmin(np.average(distribution, axis=0))
poorest = distribution[:, j_poorest]

ax2.plot(steps, richest)
ax2.plot(steps, poorest)

plt.show()

