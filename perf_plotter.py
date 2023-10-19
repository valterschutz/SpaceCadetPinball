import sys
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

model_name = sys.argv[1]

with open(f"pickles/perf_{model_name}.pkl", "rb") as file:
    scores = pickle.load(file)

scores = [score for score in scores if score > 10]
episodes = list(range(1,len(scores)+1))

fig, axes = plt.subplots(1,2)
sns.lineplot(x=episodes, y=scores, ax=axes[0])
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Agent score")
axes[0].set_title(f"Mean score: {int(sum(scores)/len(scores))}")
sns.histplot(np.log10(scores), kde=True, ax=axes[1], label="Agent score")
axes[1].axvspan(np.log10(2_500_000), np.log10(3_700_000), alpha=0.5, color='gray', label="Human score")
axes[1].set_xlabel("log10(score)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Score histogram")
axes[1].legend()
fig.tight_layout()
plt.show()
