import numpy as np
import matplotlib.pyplot as plt

base_score = np.load("base_score.npy")
score_decreases = np.load("score_decreases.npy")
feature_importances = np.mean(score_decreases, axis=0)
#np.save("feature_importance.npy", feature_importances)
print(base_score)
print(score_decreases)
print(feature_importances)

#plt.plot(feature_importances)
plt.bar(range(20),feature_importances)
plt.savefig("FI.png")

blocks = [sum(feature_importances[0:5]), 
sum(feature_importances[5:10]),
sum(feature_importances[10:15]),
sum(feature_importances[15:20])
]
plt.figure()
plt.bar(range(4), blocks)
plt.savefig("FI_blocks.png")
