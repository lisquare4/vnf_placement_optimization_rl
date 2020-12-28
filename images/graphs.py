import matplotlib.pyplot as plot

import numpy as np

import csv


def render_g1(data, name, run_idx):
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
    lg =[
        'reward',
        'penalty',
        'minibatch_loss',
    ]
    plot.legend([l + str(run_idx) for l in lg])
    # plot.yscale('log')
    plot.suptitle(name + str(run_idx))
    plot.show()

def render_g1_all(dataset, name, num_run, line_class):
    """
    render all g1 lines in to ONE graph
    """
    lgs = []
    colors = [
        'c', 'm', 'y', 'k', 'r'
    ]
    styles = [
       'None', '-', '--', '-.', ':'
    ]
    for run_idx in range(num_run):
        # 1. map same params result into same buckets

        batch_list = []
        reward_list = []
        penalty_list = []
        minibatchloss_list = []
        for row in dataset[run_idx]:
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

        if line_class == 'reward':
            plot.plot(batch_list, reward_list,
                      color=colors[run_idx], lw=1.)
        elif line_class == 'penalty':
            plot.plot(batch_list, penalty_list,
                      color=colors[run_idx], lw=1.)
        elif line_class == 'minibatch_loss':
            plot.plot(batch_list, minibatchloss_list,
                      color=colors[run_idx], lw=1.)
        else:
            print("[ERR] Fail to selector plot line class")

        lg =[ line_class ]
        lg = [l + str(run_idx) for l in lg]
        lgs.extend(lg)

    plot.legend(lgs)
    plot.suptitle(name + 'all')
    plot.show()


def run_g1(path, name, run_idx):
    with open(path + name + str(run_idx) + "/learning_history.csv") as infile:
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


        render_g1(data, name, run_idx)

def run_all_g1(path, name, num_run, line_class):

    dataset = []
    for run_idx in range(num_run):
        with open(path + name + str(run_idx) + "/learning_history.csv") as infile:
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
        dataset.append(data)


    render_g1_all(dataset, name, num_run, line_class)

def render_g2(path, name):
    pass



if __name__ == "__main__":
    path = '../save/'
    name = 'g_0.5_pe_1500_'

    RUN_ALL = False
    line_class_list = ['reward', 'penalty', 'minibatch_loss']
    if RUN_ALL:
        run_all_g1(path, name, 5, line_class_list[1])
    else:
        for run_idx in range(5):
            run_g1(path, name, run_idx)


