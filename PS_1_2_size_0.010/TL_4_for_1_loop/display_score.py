import numpy as np
import matplotlib.pyplot as plt

score = np.load("score_test.npy")
print (score)
print (score.shape)
print ("Max Test Score:", max(score))
print ("Average Test Score:", np.average(score))

plt.hist(score)
plt.savefig("hist_score.png")

