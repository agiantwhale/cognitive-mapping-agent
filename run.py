import numpy as np
import environment
import expert


def main():
    env = environment.get_game_environment()
    exp = expert.Expert()

    terminal = False
    while not terminal:
        _, info = env.observations()
        action = np.argmax(exp.get_optimal_action(info))
        _, _, terminal, _ = env.step(action)


if __name__ == '__main__':
    main()
