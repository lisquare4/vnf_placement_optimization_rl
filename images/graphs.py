import matplotlib.pyplot as plot
plot.rcParams.update({'font.size': 16})
import numpy as np
import math

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

    x = np.divide(batch_list, 1000.)
    y = minibatchloss_list/1000.
    # subplot
    # plot.plot(batch_list, reward_list, color='red', marker=">")
    # plot.plot(batch_list, penalty_list, color='green', marker="o")
    plot.plot(x, y, color='green', marker="*")
    lg =[
        # 'reward',
        # 'penalty',
        # 'minibatch_loss',
        'FSCO',
    ]
    plot.legend(lg)
    # plot.legend([l + str(run_idx) for l in lg])
    # plot.yscale('log')
    plot.ylabel('Loss function')
    plot.xlabel('Training epochs')
    plot.xticks(np.arange(min(x), max(x)+1, 2), fontsize=12)
    plot.yticks(fontsize=12)
    # plot.yticks(np.arange(math.floor(min(y)), math.ceil(max(y)), 1), fontsize=12)
    plot.text(max(x)+.5, min(y)-.64, '$(\\times 10^3)$', fontsize=12)
    plot.text(min(x), max(y)+.3, '$(\\times 10^2)$', fontsize=12)
    plot.xlim([min(x), max(x)+1])
    # plot.ylim([math.floor(min(y)), max(y)])
    plot.grid()
    plot.tight_layout()
    plot.savefig("../images/g1.pdf", dpi=400, bbox_inches='tight', pad_inches=0.1)
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
        plot.xticks(fontsize=12)
        plot.yticks(fontsize=12)
        plot.grid()
        fig.tight_layout()
        plot.savefig("../images/g4-1.pdf", dpi=400, bbox_inches='tight', pad_inches=0.1)
        plot.show()

    else:
        y_desc = "Network cost"
        x_lab = "SFC length"
        if mode == 'reward':
            fig, ax = plot.subplots(3,1)
            bp1 = ax[0].boxplot(seq_lists[0], showmeans=True, showfliers=False, widths=.25, patch_artist=True )
            for box in bp1['boxes']:
                box.set(facecolor='green')
                box.set(hatch='/')
            ax[0].legend(['NCO'], loc="upper left", framealpha=.5)

            plot.setp(ax[0], xticklabels=[12, 14, 16, 18])
            ax[0].tick_params(labelsize=12)
            ax[0].grid()

            ##############
            bp2 = ax[1].boxplot(seq_lists[2], showmeans=True, showfliers=False, widths=.25, patch_artist=True)
            ax[1].legend(['BAB'], loc="upper left", framealpha=.5)
            for box in bp2['boxes']:
                box.set(facecolor='lightblue')
                box.set(hatch='\\')

            plot.setp(ax[1], xticklabels=[12, 14, 16, 18])
            # plot.setp(ax[1], ylim=(0,8000))
            ax[1].tick_params(labelsize=12)
            ax[1].grid()

            ###############
            bp3=ax[2].boxplot(seq_lists[3], showmeans=True, showfliers=False, widths=.25, patch_artist=True)
            for box in bp3['boxes']:
                box.set(facecolor='cyan')
                box.set(hatch='x')
            ax[2].legend(['FSCO'], loc="upper left", framealpha=.5)
            # plot.setp(ax[2], ylim=(0,8000))

            plot.setp(ax[2], xticklabels=[12, 14, 16, 18])

            ax[2].tick_params(labelsize=12)
            ax[2].grid()
            fig.text(-.03, 0.5, y_desc, va='center', rotation='vertical')
            ax[2].set_xlabel(x_lab)

            fig.tight_layout()
            plot.savefig("../images/g4-2.pdf", dpi=400, bbox_inches='tight', pad_inches=0.1)
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

def run_g3(path, names, small_range, select_mem=False):

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
    ax.bar(X - 0.125, fl_cpu_data, color = 'darkgreen', width = 0.25,
           edgecolor = 'black', hatch = "///")
    ax.bar(X + 0.125, vnf_cpu_data, color = 'firebrick', width = 0.25,
           edgecolor = 'black', hatch = "\\\\\\")
    ax.legend(['FSCO', 'NCO'], loc="upper left")
    if select_mem:
        ax.set_ylabel('Exceeded Memory(Gb)')
    else:
        ax.set_ylabel('Exceeded CPUs(Core)')
    ax.set_xlabel('SFC length')
    ax.set_axisbelow(True)
    plot.xticks(X , ("12", "14", "16", "18"), fontsize=12)
    plot.yticks(fontsize=12)
    plot.grid()
    if select_mem:
        plot.savefig("../images/g3-2.pdf", dpi=400, bbox_inches='tight', pad_inches=0.1)
    else:
        plot.savefig("../images/g3-1.pdf", dpi=400, bbox_inches='tight', pad_inches=0.1)
    plot.show()

    if False:
        # Latency
        fl_ping_data = [l['fl_Latency'] for l in result_list]
        vnf_ping_data = [l['ave_Latency'] for l in result_list]

        fig, ax = plot.subplots()
        ax.bar(X - 0.125, fl_ping_data, color = 'darkgreen', width = 0.25,
               edgecolor='black', hatch= "///")
        ax.bar(X + 0.125, vnf_ping_data, color = 'firebrick', width = 0.25,
               edgecolor = 'black', hatch = "\\\\\\")
        ax.legend(['FSCO', 'NCO'], loc="upper right")
        # ax.set_ylabel('Exceeded CPUs')
        ax.set_ylabel('Exceeded Memory(Gb)')
        ax.set_xlabel('SFC length')
        ax.set_axisbelow(True)
        plot.xticks(X , ("12", "14", "16", "18"), fontsize=12)
        plot.yticks(fontsize=12)
        plot.grid()
        plot.savefig("../images/g3-2.pdf", dpi=400, bbox_inches='tight', pad_inches=0.1)
        plot.show()

    if not select_mem:
        # Bandwidth
        fl_bw_data = [l['fl_Bandwidth'] for l in result_list]
        vnf_bw_data = [l['ave_Bandwidth'] for l in result_list]

        fig, ax = plot.subplots()
        ax.bar(X - 0.125, fl_bw_data, color = 'darkgreen', width = 0.25,
               edgecolor = 'black', hatch = "///")
        ax.bar(X + 0.125, vnf_bw_data, color = 'firebrick', width = 0.25,
               edgecolor = 'black', hatch = "\\\\\\")
        ax.legend(['FSCO', 'NCO'], loc="upper left")
        # ax.set_ylabel('Exceeded CPUs')
        ax.set_ylabel('Exceeded Bandwidth(Gbps)')
        ax.set_xlabel('SFC length')
        ax.set_axisbelow(True)
        plot.xticks(X , ("12", "14", "16", "18"), fontsize=12)
        plot.yticks(fontsize=12)
        plot.grid()
        plot.savefig("../images/g3-3.pdf", dpi=400, bbox_inches='tight', pad_inches=0.1)
        plot.show()

def run_g2_all(path, names,  mode, select_J=False):

    dataset = []
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


    render_g2_all(dataset, names, mode, select_J)

def render_g2_all(dataset, names, line_class, select_J = False):

    lgs = []
    colors = [
        'c', 'm', 'y', 'k', 'r','g', 'b'
    ]
    styles = [
        'None', '-', '--', '-.', ':'
    ]
    xs = []
    ys = []
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
        batch_list = np.divide(batch_list, 1000.)
        reward_list = np.divide(reward_list, 100.)
        penalty_list = np.divide(penalty_list, 100.)
        minibatchloss_list = np.divide(minibatchloss_list, 100.)
        x = batch_list
        xs.extend(x)
        y = []
        if line_class == 'reward':
            y = reward_list
        elif line_class == 'penalty':
            y = penalty_list
        elif line_class == 'minibatch_loss':
            y = minibatchloss_list
        else:
            print("[ERR] Fail to selector plot line class")

        ys.extend(y)
        plot.plot(x, y,
                  color=colors[run_idx], lw=1.)
        # lg = [str(n)+ line_class for n in name_range]
        lg = [
            # '0.3 Penalty',
            'Errer-rate-based strategy',
            # '0.5 Penalty',
            # '0.3 Reward',
            'Reward-based strategy',
            # '0.5 Reward',
            'Average strategy',
            # 'J=500',
            # 'J=1000',
            # 'J=1500',
        ]
        if select_J:
            lg = [
                'J=500',
                'J=1000',
                'J=1500',
            ]
        lgs.extend(lg)

        plot.ylabel('Network cost')
        plot.xlabel('Training epochs')

    plot.legend(lgs)
    plot.xticks(np.arange(min(xs), max(xs), 1), fontsize=12)
    plot.yticks(np.arange(math.floor(min(ys)), math.ceil(max(ys)), 2), fontsize=12)
    plot.text(max(xs)-.3, min(ys)-.9, '$(\\times 10^3)$', fontsize=12)
    plot.text(min(xs), max(ys)+.1, '$(\\times 10^2)$', fontsize=12)
    plot.grid()
    plot.xlim([min(xs), max(xs)])
    plot.ylim([math.floor(min(ys)), max(ys)])
    plot.savefig("../images/g2.pdf", dpi=400, bbox_inches='tight', pad_inches=0.1)
    plot.show()

if __name__ == "__main__":
    DEBUG_G1_1 = DEBUG_G1_2 = \
        DEBUG_G2_1 = DEBUG_G2_2 = \
        DEBUG_G3_1 = DEBUG_G3_2 = \
        DEBUG_G4_1 = DEBUG_G4_2 = 0
    DEBUG_G4_1 = 1
    # DEBUG_G3_2 = 1

    if DEBUG_G1_1:
        # g1_small
        path = '../save/'
        name = 'l_24_ave_1500_'
        RUN_ALL = False
        line_class_list = ['reward', 'penalty', 'minibatch_loss']
        if RUN_ALL:
            run_all_g1(path, name, 5, line_class_list[1])
        else:
            run_g1(path, name, 0)

    if DEBUG_G1_2:
        # g1_large
        path = '../save/'
        name = 's_14_0.3_re_1500_'
        RUN_ALL = False
        line_class_list = ['reward', 'penalty', 'minibatch_loss']
        if RUN_ALL:
            run_all_g1(path, name, 5, line_class_list[1])
        else:
            run_g1(path, name, 0)

    if DEBUG_G2_2:
        path_g2 = '../save_backup/save/'
        names_group_g2 = []

        names_group_g2.extend([
            'g_0.3_pe_1500_',
            'g_0.3_re_1500_',
            'g_ave_1500_',
        ])

        run_g2_all(path_g2, names_group_g2, mode='reward')

    if DEBUG_G2_1:
        path_g2 = '../save_backup/save/'
        names_group_g2 = []

        names_group_g2.extend([
            's_ave_500_',
            's_ave_1000_',
            's_ave_1500_',
        ])

        run_g2_all(path_g2, names_group_g2, mode='reward', select_J=True)

    if DEBUG_G3_1:
        path = '../save/'
        small_range = list(range(12,20,2))
        names_g3 = [
            # 's_{}_0.3_re_1500_no_Solver_'.format(s_r) for s_r in small_range
            's_{}_ave_1500_no_Solver_'.format(s_r) for s_r in small_range
        ]
        run_g3(path, names_g3, small_range)

    if DEBUG_G3_2:
        path = '../save/'
        small_range = list(range(12,20,2))
        names_g3 = [
            's_{}_0.3_re_1500_no_Solver_'.format(s_r) for s_r in small_range
            # 's_{}_ave_1500_no_Solver_'.format(s_r) for s_r in small_range
        ]
        run_g3(path, names_g3, small_range, select_mem=True)

    if DEBUG_G4_1:
        path = '../save/'
        names = [
            's_12_0.3_re_1500_',
            # 's_12_0.3_re_1500_',
            # 's_12_ave_1500_',
            's_14_0.3_re_1500_',
            's_16_0.3_re_1500_',
            's_18_0.3_re_1500_',
        ]
        run_g4(path, names, mode='reward', layout='value')

    if DEBUG_G4_2:
        path = '../save/'
        names = [
            's_12_0.3_re_1500_',
            # 's_12_0.3_re_1500_',
            # 's_12_ave_1500_',
            's_14_0.3_re_1500_',
            's_16_0.3_re_1500_',
            's_18_0.3_re_1500_',
        ]
        run_g4(path, names, mode='reward', layout='ratio')
    # name = 's_14_0.3_re_1500_'
    # # g1_large
    # # name = 'l_24_ave_1500_'
    # names = [
    #     's_12_0.3_re_1500_',
    #     # 's_12_0.3_re_1500_',
    #     # 's_12_ave_1500_',
    #     's_14_0.3_re_1500_',
    #     's_16_0.3_re_1500_',
    #     's_18_0.3_re_1500_',
    # ]

    # run_g2(path, name, mode='reward')
    # run_g2(path, name, mode='penalty')

    # blend_range = list(np.arange(.3,.6,.1))
    # blend_range = [round(br, 2) for br in blend_range]



    # run_g4(path, names, mode='reward', layout='value')
    # run_g4(path, names, mode='reward', layout='ratio')



