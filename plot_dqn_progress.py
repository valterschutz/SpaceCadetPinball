import matplotlib.pyplot as plt
import pandas as pd

# Initialize lists to store training episodes and scores
training_episodes = []
scores = []

# Open and read the data from the file "train_data.txt"
with open("train_data.txt", "r") as file:
    for line in file:
        if "Score:" in line:
            score = int(line.split()[-1])
            scores.append(score)

# Calculate the moving average(5) of the scores
moving_avg = pd.Series(scores).rolling(window=5).mean()

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(range(4,len(moving_avg)), moving_avg[4:], label='Moving Average(5)')
plt.xlabel('Training Episode')
plt.ylabel('Moving Average Score')
plt.title('Moving Average(5) of Score vs. Training Episode')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig("figs/train_data.png")
plt.show()

