import argparse
import logging

import numpy as np
import pandas as pd


class S_t():
    def __init__(self, s_t=None):
        if s_t is None:
            self.s_t = np.random.multivariate_normal([70, 110], [[20, 15], [15, 40]])
        else:
            self.s_t = s_t

    def next_state(self, action):
        self.s_t += np.random.multivariate_normal([20*action, 20*action], [[1, 0.5], [0.5, 1]])

    def get_xt(self):
        return np.random.normal(self.s_t[0], 1)

    def get_ut(self):
        return np.random.normal(self.s_t[1], 1)


def physicians_policy(xt, ut):
    if xt <= 70 and ut <= 110:
        return 1
    return 0


def generate_trajectories(df: pd.DataFrame, trajectory_length: int, id: int):
    s_t = S_t()
    for i in range(trajectory_length):
        x_t, u_t = s_t.get_xt(), s_t.get_ut()
        action = physicians_policy(x_t, u_t)
        row = {'id': id, 't': i, 'A_t': action, 'X_t': x_t, 'U_t': u_t}
        df = df.append(row, ignore_index=True)
        s_t.next_state(action)
    return df


def main(number_of_trajectories, trajectory_length, starting_id, export_dir):
    log_file_name = f'{export_dir}/simple-trajectories-{number_of_trajectories}-{starting_id}.log'
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()]
                        )
    df = pd.DataFrame()
    for i in range(number_of_trajectories):
        df = generate_trajectories(df, trajectory_length, starting_id+i)
    logging.info("Saving results to csv")
    df.to_csv(f'{export_dir}/simple-observational-data-{number_of_trajectories}-{starting_id}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", help="number of trajectories to be generated", type=int)
    parser.add_argument("exportdir", help="path to output directory")
    parser.add_argument("--trajectoryLength", help="length of each trajectory", type=int, default=2)
    parser.add_argument("--startingId", help="length of each trajectory", type=int, default=0)
    args = parser.parse_args()

    main(args.trajectories, args.trajectoryLength, args.startingId, args.exportdir)
