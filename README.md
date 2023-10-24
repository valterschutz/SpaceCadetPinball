<!-- markdownlint-disable-file MD033 -->

# Q-learning SpaceCadetPinball
Training and evaluation of a DQN agent on the game [SpaceCadetPinball](https://github.com/k4zmu2a/SpaceCadetPinball).

## Instructions
Build the game first using `cmake .` and then `make`.

---

Train the agent using `train_agent.py`:
```
usage: train_agent.py [-h] [--gamma GAMMA] [--tau TAU] [--lr LR] [--eps_min EPS_MIN] [--eps_max EPS_MAX] [--eps_eval EPS_EVAL]                                              
                      [--eps_decay_per_episode EPS_DECAY_PER_EPISODE] [--buffer_size BUFFER_SIZE] [--batch_size BATCH_SIZE]                                                 
                      [--test_every_n_episodes TEST_EVERY_N_EPISODES] [--use_target_model USE_TARGET_MODEL] [--buffer_start BUFFER_START] [--n_frames N_FRAMES]             
                      mode name                                                                                                                                             
                                                                                                                                                                            
Train a RL agent to play pinball                                                                                                                                            
                                                                                                                                                                            
positional arguments:                                                                                                                                                       
  mode                  Whether to 'load' an old model or to create a 'new' model                                                                                           
  name                  Name of model                                                                                                                                       
                                                                                                                                                                            
options:                                                                                                                                                                    
  -h, --help            show this help message and exit                                                                                                                     
  --gamma GAMMA         Discount factor                                                                                                                                     
  --tau TAU             Target model update rate                                                                                                                            
  --lr LR               Learning rate                                                                                                                                       
  --eps_min EPS_MIN     Minimum allowed epsilon                                                                                                                             
  --eps_max EPS_MAX     Maximum allowed epsilon                                                                                                                             
  --eps_eval EPS_EVAL   Epsilon to use during evaluation of policy                                                                                                          
  --eps_decay_per_episode EPS_DECAY_PER_EPISODE                                                                                                                             
                        How much to decay epsilon by each episode                                                                                                           
  --buffer_size BUFFER_SIZE                                                                                                                                                 
                        Size of replay buffer                                                                                                                               
  --batch_size BATCH_SIZE                                                                                                                                                   
                        Batch size to use during training on replay buffer                                                                                                  
  --test_every_n_episodes TEST_EVERY_N_EPISODES                                                                                                                             
                        How many episodes to wait before evaluating the model again                                                                                         
  --use_target_model USE_TARGET_MODEL                                                                                                                                       
                        Whether to use a target model                                                                                                                       
  --buffer_start BUFFER_START                                                                                                                                               
                        How much to fill the replay buffer (in terms of batch size) before starting training                                                                
  --n_frames N_FRAMES   How many frames to wait between each action
```

Data is gathered during training and can be visualized with `plotter.py`:
```
usage: plotter.py [-h] name

Plot data gathered from a RL agent playing pinball

positional arguments:
  name        Name of model

options:
  -h, --help  show this help message and exit
```

Evaluate the agent and see how it plays using `eval_agent.py`:
```
usage: eval_agent.py [-h] [--episodes EPISODES] [--eps EPS] [--delay DELAY] [--n_frames N_FRAMES] name

Evaluate a RL agent to play pinball

positional arguments:
  name                 Name of model

options:
  -h, --help           show this help message and exit
  --episodes EPISODES  How many episodes to play
  --eps EPS            Which epsilon to use for evaluation
  --delay DELAY        How many seconds to wait between each step in the simulation
  --n_frames N_FRAMES  How many frames to wait between each action
```

## Links
- [Video example](https://github.com/valterschutz/SpaceCadetPinball/blob/master/example.webm)
- [Report](https://github.com/valterschutz/SpaceCadetPinball/blob/master/report.pdf)
- [Poster](https://github.com/valterschutz/SpaceCadetPinball/blob/master/poster.pdf)
