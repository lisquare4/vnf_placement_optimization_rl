import matplotlib.pyplot as plot

import numpy as np

import csv


def render_g1(data, name):
    """
    Render csv data into box graph
    csv format
    batch,
    """

    # 1. map same params result into same buckets

    batch_list = []
    reward_list = []
    penalty_list = []
    minibatchloss_list = []
    for row in data:
        batch_list.append(row['batch'])
        reward_list.append(row['reward'])
        penalty_list.append(row['penalty'])
        minibatchloss_list.append(row['minibatch_loss'])

    # minibatchloss_list = np.multiply(minibatchloss_list, -0.1)
    # batch_list= np.multiply(batch_list, 0.01)
    # reward_list= np.multiply(reward_list, .5)
    # penalty_list= np.multiply(penalty_list, .2)

    # 2. list sequences by params


    # subplot
    plot.plot(batch_list, reward_list, color='red', marker=">")
    plot.plot(batch_list, penalty_list, color='green', marker="o")
    plot.plot(batch_list, minibatchloss_list, color='yellow', marker="*")
    plot.legend(['reward', 'penalty', 'minibatch_loss'])
    # plot.yscale('log')
    plot.suptitle(name)
    plot.show()


if __name__ == "__main__":
    path = '../save/'
    name = '20000_re_0.5_0'
    with open(path + name + "/learning_history.csv") as infile:
        reader = csv.reader(infile, delimiter=',')
        data = []
        for row in reader:
            if row: # skip null row
                new_row = {}

                for cell in row:
                    key, value = cell.split(":")
                    key = key.strip()
                    if key == 'network_service[batch 0]' or key == "placement[batch 0]":
                        pass
                    else:
                        value = float(value)
                        new_row[key] = value

                data.append(new_row)


        render_g1(data, name)

