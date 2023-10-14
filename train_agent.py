
def evaluate_policy(agent, eps):
    """Evaluate the agent without training it for one episode."""
    
    print(f"Evaluating policy for one episode with eps={eps}") # TODO: should not be here

    env = GameEnvironment(600, 416)
    agent.reset_action_state()
    state = agent.get_state(env)
    qs = agent.model(state)
    qs_np = qs.cpu().numpy()[0]
    print(f"Q-values: {qs_np}") # TODO: weird to have printing here
    done, episode_reward = False, 0
    print("Actions:")
    while not done:
        action = agent.act(env, state.unsqueeze(0), eps)
        print("RrLl!.p"[action], end="")
        state, reward = agent.step(env, action)
        episode_reward += reward
        done = env.is_done()
    print("")
    del env
    time.sleep(0.1)
        
    return episode_reward, qs_np


def train(agent, buffer, batch_size=128,
        eps_max=1, eps_min=0.0, decrease_eps_steps=1000000, test_every_episodes=50):
    """Train the agent ad infinitum with decreasing epsilon."""

    # TODO: check this whole function

    # If we are resuming a previously trained model, remember where we ended
    if agent.episodes:
        episodes = agent.episodes[-1]
    else:
        episodes = 0
    step = 0 # How many states we have seen in total across all episodes
    eps = max(eps_max - (eps_max - eps_min) * step / decrease_eps_steps, eps_min)
    acc_reward, acc_loss = 0, 0 # Accumulated reward and loss over several episodes
    counter = 0 # Keeps track of how many episodes the above variables correspond to

    done = False
    training_started = False # Only start training when buffer is sufficiently full
    while True:
        if counter and training_started:
            time.sleep(0.1)
            eval_episode_reward, eval_qs = evaluate_policy(agent, eps)
            agent.episode.append(episodes)
            agent.q.append(eval_qs)
            agent.reward.append(eval_episode_reward)
            mean_loss = acc_loss / counter
            agent.loss.append(mean_loss)
            print(f"Summary of last {test_every_episodes} episodes: Step: {step}, Mean Loss: {mean_loss:.6f}, Eps: {eps}\n")
            agent.save()
            counter, acc_loss = 0, 0

        # Run some episodes
        for _ in range(test_every_episodes):
            agent.play_one_episode()

            # Episode finished
            done = False
            del env
            time.sleep(0.1)
            print(f"   Episode {episodes} done... Total reward = {acc_reward:.3f}")
            episodes += 1

def append_to_file(data):
    """Write data to a file with filename equal to process ID."""

    pid = os.getpid()
    file_path = f"textdata/{pid}.txt"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        # Create the file if it doesn't exist
        with open(file_path, 'w'):
            pass  # This will create an empty file
    
    with open(file_path, "a") as file:
        file.write(data)

def print_model_layers(model):
    """Print the contents of all layers in a model and whether the gradients are computed."""

    for name, param in model.named_parameters():
        print(f"Layer: {name}, Requires Gradients: {param.requires_grad}")

def run_train_loop(agent):
    buffer = PrioritizedReplayBuffer(1, BUFFER_SIZE)
    train(agent, buffer, batch_size=128, eps_max=1, eps_min=0.5, decrease_eps_steps=1000000, test_every_episodes=20)

if __name__ == "__main__":
    lr = 1e-6
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        name = input("DQN agent to load: ")
        pickle_filename = f"pickles/model_{name}.pkl"
        with open(pickle_filename, "rb") as file:
            agent = pickle.load(file)
    else:
        # Create a new DQN model if not loading
        # Ask for a name
        name = input("Enter a name for new DQN agent: ")
        agent = DQN(lr=lr, name=name, gamma=0.995)
    agent.optimizer = optim.Adam(agent.model.parameters(), lr=lr)
    agent.tau = 0.05
        
    run_train_loop(agent)


