import numpy as np

score = np.load("score_valid.npy")
print ("Max Valid Score:", max(score))
print ("Location of the Max", np.argmax(score), "with totol length is", score.shape[0])
print ("Last Valid Score:",score[-1])
