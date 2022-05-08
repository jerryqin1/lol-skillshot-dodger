# lol-skillshot-dodger
Course project for CS 3892. Using deep reinforcement learning methods to train agent to dodge skillshots in a custom game.

## Files
- game.py <br>
Plays the dodging game.

- game_small.py <br>
Plays the game on a much smaller screen.

- kerasrl.py <br>
Train an agent to play the game.

- game_env.py <br>
Used by the kerasrl.py script to enable training. Implements a custom Gym environment.3892

- eval.py <br>
Evaluates model on the normal screen size.3892

- eval_small.py <br>
Evaluates model on the smaller screen size.

- versus.py <br>
Play two games at once - one for the player, one for trained model. Currently still has bugs.

- hptune.py <br>
Tune hyperparameters. Unused and untested.
