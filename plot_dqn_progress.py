import matplotlib.pyplot as plt
import pandas as pd

# Initialize lists to store training episodes and scores
training_episodes = []
qs = []

# Open and read the data from the file "train_data.txt"
with open("train_data.txt", "r") as file:
    for line in file:
        if "Q" in line:
            q = float(line.split()[-1])
            qs.append(q)

# Calculate the moving average(5) of the q-values
moving_avg = pd.Series(qs).rolling(window=5).mean()

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(range(4,len(moving_avg)), moving_avg[4:], label='Moving Average(5)')
plt.xlabel('Training Episode')
plt.ylabel('Moving Average E[return]')
plt.title('Moving Average(5) of E[return] vs. Training Episode')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig("figs/train_data.png")
plt.show()

