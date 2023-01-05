import numpy as np

score = np.load("score_test.npy")
#print (score)
print (score.shape)
print ("Max Test Score:", max(score))
print ("Average Test Score:", np.average(score))

l = score.shape[0]
print (l)
means = []

for i in range(0,l):
	mean = np.average(score[0:i+1])
	means.append(mean)

import matplotlib.pyplot as plt

plt.plot(means)
plt.savefig("means.png")

