import numpy as np

score = np.load("score_test.npy")
print (score)
print (score.shape)
print ("Max Test Score:", max(score))
print ("Min Test Score:", min(score))
print ("Average Test Score:", np.average(score))
