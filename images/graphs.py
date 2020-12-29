import matplotlib.pyplot as plot

import numpy as np

import csv

from scipy import optimize

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

    minibatchloss_list = np.multiply(minibatchloss_list, -0.1)
    # batch_list= np.multiply(batch_list, 0.01)
    # reward_list= np.multiply(reward_list, .5)
    # penalty_list= np.multiply(penalty_list, .2)

    # 2. list sequences by params


    # subplot
    # plot.plot(batch_list, reward_list, color='red', marker=">")
    # plot.plot(batch_list, penalty_list, color='green', marker="o")
    plot.plot(batch_list, minibatchloss_list, color='green', marker="*")
    lg =[
        # 'reward',
        # 'penalty',
        # 'minibatch_loss',
    ]
    # plot.legend([l + str(run_idx) for l in lg])
    # plot.yscale('log')
    plot.ylabel('Loss function')
    plot.xlabel('Training epochs')
    plot.show()

def render_g2(data, name, mode):
    """
    Render csv data into box graph
    csv format
    batch,
    """

    # 1. map same params result into same buckets

    batch_list = []
    re_list = []
    pe_list = []
    so_re_list = []
    fl_pe_list = []
    fl_re_list = []
    for row in data:
        batch_list.append(row['batch'])
        re_list.append(row['reward'])
        pe_list.append(row['penalty'])
        so_re_list.append(row['solver_reward'])
        fl_re_list.append(row['fl_reward'])
        fl_pe_list.append(row['fl_penalty'])

    # minibatchloss_list = np.multiply(minibatchloss_list, -0.1)
    # batch_list= np.multiply(batch_list, 0.01)
    # reward_list= np.multiply(reward_list, .5)
    # penalty_list= np.multiply(penalty_list, .2)

    # 2. list sequences by params
    # subplot
    lg = []
    if mode == 'reward':
        plot.plot(batch_list, re_list, color='r', )
        plot.plot(batch_list, so_re_list, color='g', )
        plot.plot(batch_list, fl_re_list, color='b', )
        lg = [
            'VNF_reward',
            'Gecode_reward',
            'FL_reward(ours)',
        ]
    elif mode =='penalty':
        plot.plot(batch_list, pe_list, color='red', )
        plot.plot(batch_list, fl_pe_list, color='green', )
        lg = [
            'VNF_penalty',
            'FL_penalty(ours)',
        ]
    else:
        print("[ERR] Fail to choose mode")

    plot.legend([l for l in lg],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0 )
    # plot.yscale('log')
    plot.suptitle(name)
    plot.tight_layout(rect=[0,0,1,1])
    plot.show()

def render_g4(result_list, mode, layout):

    # 0. Init data list

    seq_lists = [[], [], [], [], []]
    # re_list = []
    # pe_list = []
    # so_re_list = []
    # fl_re_list = []
    # fl_pe_list = []

    ratio_lists = [[], [], [], [], []]

    # re_ratio_list = []
    # pe_ratio_list = []
    # so_re_ratio_list = []
    # fl_re_ratio_list = []
    # fl_pe_ratio_list = []

    y = []
    for line in result_list:
        # 1. append data point
        tag = line[0][2:4]

        y.append(int(tag))
        for idx in range(1, len(line)):
            seq = line[idx]
            seq_lists[idx-1].append(seq)
            if idx in [1,3,4]:
                ratio = 1. - np.count_nonzero(line[idx]) / len(line[idx])
                ratio *= 100.
            else:
                ratio = np.count_nonzero(line[idx]) / len(line[idx])
            ratio_lists[idx-1].append(ratio)


    # 2. list sequences by params

    if layout == 'ratio':
        fig, ax = plot.subplots(1,1)
        if mode == 'reward':
            ax.plot(y, ratio_lists[0], color='r', linestyle='--')
            ax.plot(y, ratio_lists[2], color='g', linestyle='-')
            ax.plot(y, ratio_lists[3], color='b', linestyle=':')
            ax.legend(['NCO', 'BAB', 'FSCO'],
                      loc="upper left", framealpha=.5)
        elif mode == 'penalty':
            # left
            ax.plot(y, ratio_lists[1], color='r', linestyle='--')
            ax.plot(y, ratio_lists[4], color='g', linestyle='-')
            ax.legend(['NCO', 'BAB', 'FSCO'],
                      loc="upper left", framealpha=.5)

        y_lab = "Placement error ratio(%)"
        plot.setp(ax, ylabel=y_lab)
        x_lab = "SFC length"
        plot.setp(ax, xlabel=x_lab)

        fig.tight_layout()
        plot.show()

    else:
        y_desc = "Network cost"
        if mode == 'reward':
            fig, ax = plot.subplots(3,1)
            ax[0].boxplot(seq_lists[0], showmeans=True, showfliers=False, widths=.25)
            ax[0].legend(['NCO'], loc="upper left", framealpha=.5)
            y_lab = y_desc
            plot.setp(ax[0], ylabel=y_lab)

            x_lab = "SFC length"
            plot.setp(ax[0], xlabel=x_lab)
            plot.setp(ax[0], xticklabels=[12, 14, 16, 18])
            plot.setp(ax[0], ylim=(0,8000))

            ##############
            ax[1].boxplot(seq_lists[2], showmeans=True, showfliers=False, widths=.25)
            ax[1].legend(['BAB'], loc="upper left", framealpha=.5)
            y_lab = y_desc
            plot.setp(ax[1], ylabel=y_lab)

            x_lab = "SFC length"
            plot.setp(ax[1], xlabel=x_lab)
            plot.setp(ax[1], xticklabels=[12, 14, 16, 18])
            plot.setp(ax[1], ylim=(0,8000))

            ###############
            ax[2].boxplot(seq_lists[3], showmeans=True, showfliers=False, widths=.25)
            ax[2].legend(['FSCO'], loc="upper left", framealpha=.5)
            y_lab = y_desc
            plot.setp(ax[2], ylabel=y_lab)
            plot.setp(ax[2], ylim=(0,8000))

            x_lab = "SFC length"
            plot.setp(ax[2], xlabel=x_lab)
            plot.setp(ax[2], xticklabels=[12, 14, 16, 18])

            fig.tight_layout()
            plot.show()

        elif mode == 'penalty':
            fig, ax = plot.subplots()
            ax.boxplot(seq_lists[1], showmeans=True, showfliers=False, widths=.25)
            # ax.legend(['NCO'], loc="upper left", framealpha=.5)

            ax.boxplot(seq_lists[3], showmeans=True, showfliers=False, widths=.25)
            # ax.legend(['FSCO'], loc="upper left", framealpha=.5)
            y_lab = y_desc
            plot.setp(ax, ylabel=y_lab)

            x_lab = "SFC length"
            plot.setp(ax, xlabel=x_lab)
            plot.setp(ax, xticklabels=[12, 14, 16, 18])

            ##############
            fig, ax = plot.subplots()
            ax.boxplot(seq_lists[4], showmeans=True, showfliers=False, widths=.25)
            # ax.legend(['BAB'], loc="upper left", framealpha=.5)

            ax.boxplot(seq_lists[3], showmeans=True, showfliers=False, widths=.25)
            # ax.legend(['FSCO'], loc="upper left", framealpha=.5)
            y_lab = y_desc
            plot.setp(ax, ylabel=y_lab)

            x_lab = "SFC length"
            plot.setp(ax, xlabel=x_lab)
            plot.setp(ax, xticklabels=[12, 14, 16, 18])





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

def run_g2(path, name, mode):
    with open(path + name + 'test.csv') as infile:
        reader = csv.reader(infile, delimiter=',')
        data = []
        for row in reader:
            if row: # skip null row
                new_row = {}

                for cell in row:
                    key, value = cell.split(":")
                    key = key.strip()
                    if key == 'network_service' or key[-9:] == "placement":
                        pass
                    else:
                        value = float(value)
                        new_row[key] = value

                data.append(new_row)

        render_g2(data, name, mode)


def run_g4(path, names, mode, layout):

    result_list = []
    for name in names:

        # maintain list of rewards and penalties, count failed ones
        rewards = []
        penalties = []
        so_rewards = []
        fl_rewards = []
        fl_penalties = []
        with open(path + name + 'test.csv') as infile:
            reader = csv.reader(infile, delimiter=',')
            for row in reader:
                if row: # skip null row
                    new_row = {}

                    for cell in row:
                        key, value = cell.split(":")
                        key = key.strip()
                        if key == 'network_service' or key[-9:] == "placement":
                            pass
                        else:
                            value = float(value)
                            new_row[key] = value
                    rewards.append(new_row['reward'])
                    so_rewards.append(new_row['solver_reward'])
                    fl_rewards.append(new_row['fl_reward'])
                    penalties.append(new_row['penalty'])
                    fl_penalties.append(new_row['fl_penalty'])
        result_list.append([
            name, rewards, penalties, so_rewards, fl_rewards, fl_penalties
        ])

    render_g4(result_list, mode, layout)

def run_g3(path, names, small_range):

    result_list = []

    for name in names:
        new_row = {}
        with open(path + name + 'test.csv') as infile:
            reader = csv.reader(infile, delimiter=',')
            row = list(reader)
            for cell in row[0]:
                key, value = cell.split(":")
                key = key.strip()
                if key[-3:] != "CPU" and key[-7:] != "Latency" and key[-9:] != "Bandwidth":
                    pass
                else:
                    value = float(value)
                    new_row[key] = value
            result_list.append(new_row)

    for dc in result_list:
        for k,v in dc.items():
            dc[k] = v/128.
    # rendering q3
    X = np.arange(4)
    #CPU
    fl_cpu_data = [l['fl_CPU'] for l in result_list]
    vnf_cpu_data = [l['ave_CPU'] for l in result_list]

    fig, ax = plot.subplots()
    ax.bar(X + 0.00, fl_cpu_data, color = 'g', width = 0.25)
    ax.bar(X + 0.25, vnf_cpu_data, color = 'gray', width = 0.25)
    ax.legend(['FSCO', 'NCO'], loc="upper left")
    # ax.set_ylabel('Exceeded CPUs')
    ax.set_title('Exceeded CPUs(Core)')
    plot.xticks(X , ("12", "14", "16", "18"))
    plot.show()

    # Latency
    fl_ping_data = [l['fl_Latency'] for l in result_list]
    vnf_ping_data = [l['ave_Latency'] for l in result_list]

    fig, ax = plot.subplots()
    ax.bar(X + 0.00, fl_ping_data, color = 'g', width = 0.25)
    ax.bar(X + 0.25, vnf_ping_data, color = 'gray', width = 0.25)
    ax.legend(['FSCO', 'NCO'], loc="upper right")
    # ax.set_ylabel('Exceeded CPUs')
    ax.set_title('Exceeded Memory(Gb)')
    plot.xticks(X , ("12", "14", "16", "18"))
    plot.show()

    # Bandwidth
    fl_bw_data = [l['fl_Bandwidth'] for l in result_list]
    vnf_bw_data = [l['ave_Bandwidth'] for l in result_list]

    fig, ax = plot.subplots()
    ax.bar(X + 0.00, fl_bw_data, color = 'g', width = 0.25)
    ax.bar(X + 0.25, vnf_bw_data, color = 'gray', width = 0.25)
    ax.legend(['FSCO', 'NCO'], loc="upper left")
    # ax.set_ylabel('Exceeded CPUs')
    ax.set_title('Exceeded Bandwidth(Gbps)')
    plot.xticks(X , ("12", "14", "16", "18"))
    plot.show()

def run_g2_all(path, groups, name_range, mode):

    dataset = []
    for group in groups:
        names = group
        for name_idx in range(len(names)):
            with open(path + names[name_idx]+ "0" + "/learning_history.csv") as infile:
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


    render_g2_all(dataset, names, name_range, mode)

def render_g2_all(dataset, names, name_range, line_class):

    lgs = []
    colors = [
        'c', 'm', 'y', 'k', 'r'
    ]
    styles = [
        'None', '-', '--', '-.', ':'
    ]
    for run_idx in range(len(names)):
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

        lg = [str(n)+ line_class for n in name_range]
        lgs.extend(lg)

    plot.legend(lgs)
    plot.show()

if __name__ == "__main__":
    path = '../save/'
    name = 's_12_0.3_re_1500_'
    names = [
        's_12_0.3_re_1500_',
        # 's_12_0.3_re_1500_',
        # 's_12_ave_1500_',
        's_14_0.3_re_1500_',
        's_16_0.3_re_1500_',
        's_18_0.3_re_1500_',
    ]

    # run_g2(path, name, mode='reward')
    # run_g2(path, name, mode='penalty')

    blend_range = list(np.arange(.3,.6,.1))
    blend_range = [round(br, 2) for br in blend_range]

    path_g2 = '../save_backup/save/'
    names_group_g2 =[]
    names_group_g2.extend([
        'g_{}_pe_1000_'.format(br) for br in blend_range
        # 's_{}_ave_1500_no_Solver_'.format(s_r) for s_r in small_range
    ])
    names_group_g2.extend([
        'g_{}_re_1000_'.format(br) for br in blend_range
        # 's_{}_ave_1500_no_Solver_'.format(s_r) for s_r in small_range
    ])
    names_group_g2.extend(['g_ave_1000_'])
    run_g2_all(path_g2, names_g2, blend_range, mode='reward')


    # run_g4(path, names, mode='reward', layout='value')
    # run_g4(path, names, mode='reward', layout='ratio')

    # RUN_ALL = False
    # line_class_list = ['reward', 'penalty', 'minibatch_loss']
    # if RUN_ALL:
    #     run_all_g1(path, name, 5, line_class_list[1])
    # else:
    #     for run_idx in range(5):
    #         run_g1(path, name, run_idx)


    # small_range = list(range(12,20,2))
    # names_g3 = [
    #     # 's_{}_0.3_re_1500_no_Solver_'.format(s_r) for s_r in small_range
    #     's_{}_ave_1500_no_Solver_'.format(s_r) for s_r in small_range
    # ]
    # run_g3(path, names_g3, small_range)

