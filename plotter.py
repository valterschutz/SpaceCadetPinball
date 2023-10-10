import argparse
import re
import matplotlib.pyplot as plt

def extract_q_values_and_mean_loss(file_path):
    q_values = []
    mean_loss_values = []

    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expressions to find Q-values and mean loss
    q_pattern = re.compile(r'Q values at game start: \[([\d\.\s\-]+)\]', re.MULTILINE)
    mean_loss_pattern = re.compile(r'Summary of last 15 episodes: .* Mean Loss: (\d+\.\d+),', re.MULTILINE)

    q_matches = q_pattern.findall(content)
    mean_loss_matches = mean_loss_pattern.findall(content)

    for q_match, mean_loss_match in zip(q_matches, mean_loss_matches):
        q_values.extend([float(val) for val in q_match.split()])
        mean_loss_values.append(float(mean_loss_match))

    return q_values, mean_loss_values

def plot_q_values(q_values):
    plt.figure()
    for i in range(7):
        plt.plot(range(i, len(q_values), 7), q_values[i::7], label=f'Q{i+1}')

    plt.xlabel('Episode')
    plt.ylabel('Q-Values')
    plt.legend()
    plt.savefig('figs/q-values.png')
    plt.close()

def plot_mean_loss(mean_loss_values):
    print(mean_loss_values)
    plt.figure()
    plt.plot(mean_loss_values)
    plt.xlabel('Episode')
    plt.ylabel('Mean Loss')
    plt.savefig('figs/mean-loss.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and plot Q-values and mean loss from a text file.')
    parser.add_argument('file_path', type=str, help='Path to the input text file')
    args = parser.parse_args()

    q_values, mean_loss_values = extract_q_values_and_mean_loss(args.file_path)
    plot_q_values(q_values)
    plot_mean_loss(mean_loss_values)


