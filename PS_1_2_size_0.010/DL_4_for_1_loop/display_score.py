import numpy as np
import matplotlib.pyplot as plt

score = np.load("score_test.npy")

score[513] = -0.3863719308557192
np.save("score_test1.npy", score)
#print (score)
print (score.shape)
print ("Max Test Score:", max(score))
print ("Min Test Score:", min(score))
print ("Average Test Score:", np.average(score))

plt.hist(score)
plt.savefig("hist_score.png")


count = 0
count1 = 0

for i in range(len(score)):
	if score[i] <-3 : print(i)
	if score[i] <=0:
		count = count + 1
	else:
		count1 = count1 + 1
print(count)
print(count1)


