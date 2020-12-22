#-*- coding: utf-8 -*-
"""
    Placement using Neural Combinational Optimization.

    Author: Ruben Solozabal, PhD student at the University of the Basque Country (UPV-EHU) - Bilbao
    Date: June 2019
"""
import logging
import tensorflow as tf
from environment import *
from service_batch_generator import *
from agent import *
from config import *
from solver import *
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import csv
import os
from first_fit import *
import math

""" Globals """
DEBUG = True
FL_DEBUG = True
TEST_VNF_FL = True
TEST_VNF = True

def print_trainable_parameters():
    """ Calculate the number of weights """

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print('shape: ', shape, 'variable_parameters: ', variable_parameters)
        total_parameters += variable_parameters
    print('Total_parameters: ', total_parameters)


def calculate_reward(env, networkServices, placement, num_samples, agent):
    """ Evaluate the batch of states into the environmnet """

    lagrangian = np.zeros(config.batch_size)
    penalty = np.zeros(config.batch_size)
    reward = np.zeros(config.batch_size)
    constraint_occupancy = np.zeros(config.batch_size)
    constraint_bandwidth = np.zeros(config.batch_size)
    constraint_latency = np.zeros(config.batch_size)

    reward_sampling = np.zeros(num_samples)
    constraint_occupancy_sampling = np.zeros(num_samples)
    constraint_bandwidth_sampling = np.zeros(num_samples)
    constraint_latency_sampling = np.zeros(num_samples)

    indices = np.zeros(config.batch_size)

    # Compute environment
    for batch in range(config.batch_size):
        for sample in range(num_samples):
            env.clear()
            env.step(networkServices.service_length[batch], networkServices.state[batch], placement[sample][batch])
            reward_sampling[sample] = env.reward
            constraint_occupancy_sampling[sample] = env.constraint_occupancy
            constraint_bandwidth_sampling[sample] = env.constraint_bandwidth
            constraint_latency_sampling[sample] = env.constraint_latency

        penalty_sampling = agent.lambda_occupancy * constraint_occupancy_sampling + agent.lambda_bandwidth * constraint_bandwidth_sampling + agent.lambda_latency * constraint_latency_sampling
        lagrangian_sampling = reward_sampling + penalty_sampling

        index = np.argmin(lagrangian_sampling)

        lagrangian[batch] = lagrangian_sampling[index]
        penalty[batch] = penalty_sampling[index]
        reward[batch] = reward_sampling[index]

        constraint_occupancy[batch] = constraint_occupancy_sampling[index]
        constraint_bandwidth[batch] = constraint_bandwidth_sampling[index]
        constraint_latency[batch] = constraint_latency_sampling[index]

        indices[batch] = index

    return lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, indices

def session_preparation():
    """perarations for sessions, reusable for all iterations"""

    """ Log """
    logging.basicConfig(level=logging.DEBUG)  # filename='example.log'
    # DEBUG, INFO, WARNING, ERROR, CRITICAL

    """ Environment """
    env = Environment(config.num_cpus, config.num_vnfd, config.env_profile)

    """ Network service generator """
    vocab_size = config.num_vnfd + 1
    networkServices = ServiceBatchGenerator(config.batch_size, config.min_length, config.max_length, vocab_size)

    """ Agent """
    agent = Agent(config)

    """ Configure Saver to save & restore model variables """
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, max_to_keep=None)

    return env, networkServices, agent, saver


def learning(sess, config, env, networkServices, agent, saver):
    """learning"""

    """nest function learning at a certain cycle and iteration of run"""
    def learning_cycle_run(cycle_idx, run_idx, tf_variables):
        """
              cycle
        """
        epoch_cycle_start = cycle_idx * cycle_iters
        epoch_cycle_end = epoch_cycle_start + cycle_iters
        if epoch_cycle_start + cycle_iters * 2 > config.num_epoch:
            epoch_cycle_end = config.num_epoch

        save_to = config.save_to + str(run_idx)
        # Restore model
        if cycle_idx> 0 and tf_variables:
            #TODO assign tf_variables to model
            ops = [] # serie of ops to update global variables from old to new
            tf_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
            for item_idx in range(len(tf_variables)):
                ops.append(tf_to_save[item_idx].assign(tf_variables[item_idx]))
            sess.run(ops)

            print("\nModel restored from global variables.")

        # Summary writer
        writer = tf.summary.FileWriter("summary/repo", sess.graph)

        if config.save_model:
            filePath = "{}/learning_history.csv".format(save_to)

            if not os.path.exists(os.path.dirname(filePath)):
                os.makedirs(os.path.dirname(filePath))

            # if os.path.exists(filePath) and not config.load_model:
            #     os.remove(filePath)

        print("\nStart learning...")

        try:
            episode = 0
            _sum_penalty = 0
            _sum_reward = 0
            _sum_loss_rl = 0
            for episode in range(epoch_cycle_start, epoch_cycle_end):

                # New batch of states
                networkServices.getNewState()

                # Mask
                mask = np.zeros((config.batch_size, config.max_length))
                for i in range(config.batch_size):
                    for j in range(networkServices.service_length[i], config.max_length):
                        mask[i, j] = 1

                # RL Learning
                feed = {agent.input_: networkServices.state,
                        agent.input_len_: [item for item in networkServices.service_length],
                        agent.mask: mask}

                # Run placement
                placement, decoder_softmax, _, baseline = sess.run(
                    [agent.actor.decoder_exploration, agent.actor.decoder_softmax, agent.actor.attention_plot,
                     agent.valueEstimator.value_estimate], feed_dict=feed)
                # positions, attention_plot = sess.run([agent.actor.positions, agent.actor.attention_plot], feed_dict=feed)

                # Interact with the environment to return reward
                lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, indices = calculate_reward(
                    env, networkServices, placement, 1, agent)

                placement_ = np.zeros((config.batch_size, config.max_length))
                for batch in range(config.batch_size):
                    placement_[batch] = placement[int(indices[batch])][batch]

                feed = {agent.placement_holder: placement_,
                        agent.input_: networkServices.state,
                        agent.input_len_: [item for item in networkServices.service_length],
                        agent.mask: mask,
                        agent.baseline_holder: baseline,
                        agent.lagrangian_holder: [item for item in lagrangian]}

                # Update our value estimator
                feed_dict_ve = {agent.input_: networkServices.state,
                                agent.valueEstimator.target: lagrangian}

                _, loss = sess.run([agent.valueEstimator.train_op, agent.valueEstimator.loss], feed_dict_ve)

                # Update actor
                summary, _, loss_rl = sess.run([agent.merged, agent.train_step, agent.loss_rl], feed_dict=feed)

                # accumulate metrics
                _sum_penalty += np.mean(penalty)
                _sum_reward += np.mean(reward)
                _sum_loss_rl += np.mean(loss_rl)

                # Print learning
                if episode == 0 or episode % 100 == 0:
                    print("------------")
                    print("Episode: ", episode)
                    print("Minibatch loss: ", loss_rl)
                    print("Network service[batch0]: ", networkServices.state[0])
                    print("Len[batch0]", networkServices.service_length[0])
                    print("Placement[batch0]: ", placement_[0])

                    # agent.actor.plot_attention(attention_plot[0])
                    # print("prob:", decoder_softmax[0][0])
                    # print("prob:", decoder_softmax[0][1])
                    # print("prob:", decoder_softmax[0][2])

                    print("Baseline[batch0]: ", baseline[0])
                    print("Reward[batch0]: ", reward[0])
                    print("Penalty[batch0]: ", penalty[0])
                    print("Lagrangian[batch0]: ", lagrangian[0])

                    print("Value Estimator loss: ", np.mean(loss))
                    print("Mean penalty: ", np.mean(penalty))
                    print("Count_nonzero: ", np.count_nonzero(penalty))

                if episode % 10 == 0:
                    # Save in summary
                    writer.add_summary(summary, episode)

                if config.save_model and (episode == 0 or episode % 100 == 0):
                    # Save in csv
                    csvData = ['batch: {}'.format(episode),
                               ' network_service[batch 0]: {}'.format(networkServices.state[0]),
                               ' placement[batch 0]: {}'.format(placement_[0]),
                               ' reward: {}'.format(np.mean(reward)),
                               ' lagrangian: {}'.format(np.mean(lagrangian)),
                               ' baseline: {}'.format(np.mean(baseline)),
                               ' advantage: {}'.format(np.mean(lagrangian) - np.mean(baseline)),
                               ' penalty: {}'.format(np.mean(penalty)),
                               ' minibatch_loss: {}'.format(loss_rl)]

                    filePath = "{}/learning_history.csv".format(save_to)
                    with open(filePath, 'a') as csvFile:
                        writer2 = csv.writer(csvFile)
                        writer2.writerow(csvData)

                    csvFile.close()

                # Save intermediary model variables
                if config.save_model and episode % max(1, epoch_cycle_end - epoch_cycle_start) == 0 and episode != 0:
                    save_path = saver.save(sess, "{}/tmp.ckpt".format(save_to), global_step=episode)
                    print("\nModel saved in file: %s" % save_path)

                episode += 1


            print("\nLearning COMPLETED!")

        except KeyboardInterrupt:
            print("\nLearning interrupted by user.")

        # Save model
        if config.save_model:
            save_path = saver.save(sess, "./{}/tf_placement.ckpt".format(save_to))
            print("\nModel saved in file: %s" % save_path)

        # current model variables_to_save
        variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
        values_to_save = [v.eval() for v in variables_to_save]

        # calculate metrics:
        #       penalty, reward, minibatch_loss
        metrics = [_sum_reward, _sum_penalty, _sum_loss_rl]

        return values_to_save, metrics

    """
        cycle 5 times, each time contains 5 runs
    """
    # running in cycles, cycle = 5
    num_runs = config.num_runs
    cycle_iters = config.cycle_iters
    num_cycle = math.ceil(config.num_epoch / cycle_iters)
    weight_ave = None

    for cycle_idx in range(num_cycle):
        # 1. Execute a full cycle
        weight_list = []
        _weights = []
        _metrics = []

        for run_idx in range(num_runs):
            _weights, _metrics = learning_cycle_run(cycle_idx, run_idx, weight_ave)
            weight_list.append(_weights)

        # 2. FL average model parameters(weights)

        # 2.1 weight selector/map
        weight_map = [0]*21
        if config.weight_map == 'actor_all':
            weight_map[0:12] = [1]*12
        elif config.weight_map == 'value_estimator':
            weight_map[12:19] = [1]*7
        elif config.weight_map == 'rl':
            weight_map[19:] = [1]*2

        # 2.2 calculate default(ave all result)
        weight_sum = weight_list[0]
        for weight_run_idx in range(1, len(weight_list)):
            for item_idx in range(len(weight_sum)):
                weight_sum[item_idx] = np.add(weight_sum[item_idx], weight_list[weight_run_idx][item_idx])
        weight_ave = [np.divide(i, num_runs) for i in weight_sum]

        # 2.3 apply weights to latest run with trend

        best_reward_idx = 0
        best_penalty_idx= 0
        best_loss_idx = 0

        _best_reward = _metrics[0][0]
        _best_penalty = _metrics[0][1]
        _best_loss = _metrics[0][2]

        for idx in range(_metrics):
            # reward
            if _best_reward < _metrics[idx][0]:
                _best_reward = _metrics[idx][0]
                best_reward_idx = idx

            # penalty
            if _best_penalty < _metrics[idx][1]:
                _best_penalty = _metrics[idx][1]
                best_penalty_idx = idx

            # loss
            if _best_loss < _metrics[idx][2]:
                _best_loss = _metrics[idx][2]
                best_loss_idx = idx

        if config.trend_mode == "reward":
            best_reward_coef = config.trend_coef
            for item_idx in range(len(weight_ave)):
                if weight_map[item_idx]:
                    weight_ave[item_idx] = weight_list[best_reward_idx] * best_reward_coef
        elif config.trend_mode == "penalty":
            best_penalty_coef = config.trend_coef
            for item_idx in range(len(weight_ave)):
                if weight_map[item_idx]:
                    weight_ave[item_idx] = weight_list[best_penalty_idx] * best_penalty_coef
        elif config.trend_mode == "loss":
            best_loss_coef = config.trend_coef
            for item_idx in range(len(weight_ave)):
                if weight_map[item_idx]:
                    weight_ave[item_idx] = weight_list[best_loss_idx] * best_loss_coef
        else:
            print("[FAIL] Weight trend selection failed")

def inference(sess, config, env, networkServices, agent, saver):
    """
        Inference
    """

    # New batch of states
    networkServices.getNewState()

    # Mask
    mask = np.zeros((config.batch_size, config.max_length))
    for i in range(config.batch_size):
        for j in range(networkServices.service_length[i], config.max_length):
            mask[i, j] = 1

    """
        TEST VNF FL
    """
    fl_reward_t = []
    fl_placement_t = []
    fl_penalty_t = []
    fl_constraint_occupancy_t = []
    fl_constraint_bandwidth_t = []
    fl_constraint_latency_t = []

    if TEST_VNF_FL:
    # Count the number of models "m" used in the active search: model_1, model_2 ...
        m = 0
        while os.path.exists("{}_{}".format(config.fl_load_from, m + 1)):  m += 1

        placement_m = [[]] * m
        placement_temp_m = [[]] * m

        penalty_m = [0] * m
        penalty_temp_m = [0] * m

        lagrangian_m = [0] * m
        lagrangian_temp_m = [0] * m

        reward_m = [0] * m
        reward_temp_m = [0] * m

        constraint_occupancy_m = [0] * m
        constraint_occupancy_temp_m = [0] * m

        constraint_bandwidth_m = [0] * m
        constraint_bandwidth_temp_m = [0] * m

        constraint_latency_m = [0] * m
        constraint_latency_temp_m = [0] * m

        for i in range(m):
            # Restore variables from disk

            saver.restore(sess, "{}_{}/tf_placement.ckpt".format(config.fl_load_from, i + 1))
            print("Model restored.")

            # Compute placement
            feed = {agent.input_: networkServices.state,
            agent.input_len_: [item for item in networkServices.service_length], agent.mask: mask}

            placement_temp, placement, decoder_softmax_temp, decoder_softmax = sess.run \
            ([agent.actor.decoder_sampling, agent.actor.decoder_prediction, agent.actor.decoder_softmax_temp,
            agent.actor.decoder_softmax], feed_dict=feed)

            # Interact with the environment with greedy placement
            lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, _ = calculate_reward(
            env, networkServices, placement, 1, agent)

            # Interact with the environment with sampling technique
            lagrangian_temp, penalty_temp, reward_temp, constraint_occupancy_temp, constraint_bandwidth_temp, constraint_latency_temp, indices = calculate_reward(
            env, networkServices, placement_temp, agent.actor.samples, agent)

            # Store the output of each model
            placement_m[i] = placement
            for batch in range(config.batch_size):
                placement_temp_m[i].append(placement_temp[int(indices[batch])][batch])

            penalty_m[i] = penalty
            penalty_temp_m[i] = penalty_temp

            print("Errors model ", i, ":", np.count_nonzero(penalty_m[i]), "/", config.batch_size)
            print("Errors model temperature ", i, ":", np.count_nonzero(penalty_temp_m[i]), "/", config.batch_size)

            lagrangian_m[i] = lagrangian
            lagrangian_temp_m[i] = lagrangian_temp

            reward_m[i] = reward
            reward_temp_m[i] = reward_temp

            constraint_occupancy_m[i] = constraint_occupancy
            constraint_occupancy_temp_m[i] = constraint_occupancy_temp

            constraint_bandwidth_m[i] = constraint_bandwidth
            constraint_bandwidth_temp_m[i] = constraint_bandwidth_temp

            constraint_latency_m[i] = constraint_latency
            constraint_latency_temp_m[i] = constraint_latency_temp

        penalty_m = np.vstack(penalty_m)
        penalty_temp_m = np.vstack(penalty_temp_m)

        lagrangian_m = np.stack(lagrangian_m)
        lagrangian_temp_m = np.stack(lagrangian_temp_m)

        reward_m = np.stack(reward_m)
        reward_temp_m = np.stack(reward_temp_m)

        constraint_occupancy_m = np.stack(constraint_occupancy_m)
        constraint_occupancy_temp_m = np.stack(constraint_occupancy_temp_m)

        constraint_bandwidth_m = np.stack(constraint_bandwidth_m)
        constraint_bandwidth_temp_m = np.stack(constraint_bandwidth_temp_m)

        constraint_latency_m = np.stack(constraint_latency_m)
        constraint_latency_temp_m = np.stack(constraint_latency_temp_m)

        index = []

        best_placement = []
        best_placement_t = []
        best_lagrangian = []
        best_lagrangian_t = []
        best_penalty = []
        best_penalty_t = []
        best_reward = []
        best_reward_t = []
        best_constraint_occupancy = []
        best_constraint_occupancy_t = []
        best_constraint_bandwidth = []
        best_constraint_bandwidth_t = []
        best_constraint_latency = []
        best_constraint_latency_t = []

        # Calculate and store the best model
        for batch in range(config.batch_size):
            index_l = np.argmin([row[batch] for row in lagrangian_m])
            index_p = np.argmin([row[batch] for row in penalty_m])

            assert penalty_m[index_l][batch] <= penalty_m[index_p][batch]

            best_placement.append(placement_m[index_l][0][batch])
            best_lagrangian.append(lagrangian_m[index_l][batch])
            best_penalty.append(penalty_m[index_l][batch])
            best_reward.append(reward_m[index_l][batch])
            best_constraint_occupancy.append(constraint_occupancy_m[index_l][batch])
            best_constraint_bandwidth.append(constraint_bandwidth_m[index_l][batch])
            best_constraint_latency.append(constraint_latency_m[index_l][batch])

            # Temperature

            index_lt = np.argmin([row[batch] for row in lagrangian_temp_m])
            index_pt = np.argmin([row[batch] for row in penalty_temp_m])

            best_placement_t.append(placement_temp_m[index_l][batch])
            best_lagrangian_t.append(lagrangian_temp_m[index_l][batch])
            best_penalty_t.append(penalty_temp_m[index_l][batch])
            best_reward_t.append(reward_temp_m[index_l][batch])
            best_constraint_occupancy_t.append(constraint_occupancy_temp_m[index_l][batch])
            best_constraint_bandwidth_t.append(constraint_bandwidth_temp_m[index_l][batch])
            best_constraint_latency_t.append(constraint_latency_temp_m[index_l][batch])

        fl_reward_t = best_reward_t
        fl_placement_t = best_placement_t
        fl_penalty_t = best_penalty_t
        fl_constraint_occupancy_t = best_constraint_occupancy_t
        fl_constraint_bandwidth_t = best_constraint_bandwidth_t
        fl_constraint_latency_t = best_constraint_latency_t
        print("Total errors: ", np.count_nonzero(best_penalty), "/", config.batch_size)
        print("Total errors temperature: ", np.count_nonzero(best_penalty_t), "/", config.batch_size)

    """
        TEST VNF
    """

    best_reward_t = []
    best_placement_t = []
    best_penalty_t = []
    best_constraint_occupancy_t = []
    best_constraint_bandwidth_t = []
    best_constraint_latency_t = []

    if TEST_VNF:
        # Count the number of models "m" used in the active search: model_1, model_2 ...
        m = 0
        while os.path.exists("{}_{}".format(config.load_from, m + 1)):  m += 1

        placement_m = [[]] * m
        placement_temp_m = [[]] * m

        penalty_m = [0] * m
        penalty_temp_m = [0] * m

        lagrangian_m = [0] * m
        lagrangian_temp_m = [0] * m

        reward_m = [0] * m
        reward_temp_m = [0] * m

        constraint_occupancy_m = [0] * m
        constraint_occupancy_temp_m = [0] * m

        constraint_bandwidth_m = [0] * m
        constraint_bandwidth_temp_m = [0] * m

        constraint_latency_m = [0] * m
        constraint_latency_temp_m = [0] * m

        for i in range(m):
            # Restore variables from disk

            saver.restore(sess, "{}_{}/tf_placement.ckpt".format(config.load_from, i + 1))
            print("Model restored.")

            # Compute placement
            feed = {agent.input_: networkServices.state,
                    agent.input_len_: [item for item in networkServices.service_length], agent.mask: mask}

            placement_temp, placement, decoder_softmax_temp, decoder_softmax = sess.run \
                ([agent.actor.decoder_sampling, agent.actor.decoder_prediction, agent.actor.decoder_softmax_temp,
                  agent.actor.decoder_softmax], feed_dict=feed)

            # Interact with the environment with greedy placement
            lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, _ = calculate_reward(
                env, networkServices, placement, 1, agent)

            # Interact with the environment with sampling technique
            lagrangian_temp, penalty_temp, reward_temp, constraint_occupancy_temp, constraint_bandwidth_temp, constraint_latency_temp, indices = calculate_reward(
                env, networkServices, placement_temp, agent.actor.samples, agent)

            # Store the output of each model
            placement_m[i] = placement
            for batch in range(config.batch_size):
                placement_temp_m[i].append(placement_temp[int(indices[batch])][batch])

            penalty_m[i] = penalty
            penalty_temp_m[i] = penalty_temp

            print("Errors model ", i, ":", np.count_nonzero(penalty_m[i]), "/", config.batch_size)
            print("Errors model temperature ", i, ":", np.count_nonzero(penalty_temp_m[i]), "/", config.batch_size)

            lagrangian_m[i] = lagrangian
            lagrangian_temp_m[i] = lagrangian_temp

            reward_m[i] = reward
            reward_temp_m[i] = reward_temp

            constraint_occupancy_m[i] = constraint_occupancy
            constraint_occupancy_temp_m[i] = constraint_occupancy_temp

            constraint_bandwidth_m[i] = constraint_bandwidth
            constraint_bandwidth_temp_m[i] = constraint_bandwidth_temp

            constraint_latency_m[i] = constraint_latency
            constraint_latency_temp_m[i] = constraint_latency_temp

        penalty_m = np.vstack(penalty_m)
        penalty_temp_m = np.vstack(penalty_temp_m)

        lagrangian_m = np.stack(lagrangian_m)
        lagrangian_temp_m = np.stack(lagrangian_temp_m)

        reward_m = np.stack(reward_m)
        reward_temp_m = np.stack(reward_temp_m)

        constraint_occupancy_m = np.stack(constraint_occupancy_m)
        constraint_occupancy_temp_m = np.stack(constraint_occupancy_temp_m)

        constraint_bandwidth_m = np.stack(constraint_bandwidth_m)
        constraint_bandwidth_temp_m = np.stack(constraint_bandwidth_temp_m)

        constraint_latency_m = np.stack(constraint_latency_m)
        constraint_latency_temp_m = np.stack(constraint_latency_temp_m)

        index = []

        best_placement = []
        best_placement_t = []
        best_lagrangian = []
        best_lagrangian_t = []
        best_penalty = []
        best_penalty_t = []
        best_reward = []
        best_reward_t = []
        best_constraint_occupancy = []
        best_constraint_occupancy_t = []
        best_constraint_bandwidth = []
        best_constraint_bandwidth_t = []
        best_constraint_latency = []
        best_constraint_latency_t = []

        # Calculate and store the best model
        for batch in range(config.batch_size):
            index_l = np.argmin([row[batch] for row in lagrangian_m])
            index_p = np.argmin([row[batch] for row in penalty_m])

            assert penalty_m[index_l][batch] <= penalty_m[index_p][batch]

            best_placement.append(placement_m[index_l][0][batch])
            best_lagrangian.append(lagrangian_m[index_l][batch])
            best_penalty.append(penalty_m[index_l][batch])
            best_reward.append(reward_m[index_l][batch])
            best_constraint_occupancy.append(constraint_occupancy_m[index_l][batch])
            best_constraint_bandwidth.append(constraint_bandwidth_m[index_l][batch])
            best_constraint_latency.append(constraint_latency_m[index_l][batch])

            # Temperature

            index_lt = np.argmin([row[batch] for row in lagrangian_temp_m])
            index_pt = np.argmin([row[batch] for row in penalty_temp_m])

            best_placement_t.append(placement_temp_m[index_l][batch])
            best_lagrangian_t.append(lagrangian_temp_m[index_l][batch])
            best_penalty_t.append(penalty_temp_m[index_l][batch])
            best_reward_t.append(reward_temp_m[index_l][batch])
            best_constraint_occupancy_t.append(constraint_occupancy_temp_m[index_l][batch])
            best_constraint_bandwidth_t.append(constraint_bandwidth_temp_m[index_l][batch])
            best_constraint_latency_t.append(constraint_latency_temp_m[index_l][batch])

        print("Total errors: ", np.count_nonzero(best_penalty), "/", config.batch_size)
        print("Total errors temperature: ", np.count_nonzero(best_penalty_t), "/", config.batch_size)

    # Test Gecode solver
    if config.enable_performance:

        sReward = np.zeros(config.batch_size)

        filePath = '{}_test.csv'.format(config.load_from)
        if os.path.exists(filePath):
            os.remove(filePath)

        for batch in tqdm(range(config.batch_size)):

            # hPlacement, hEnergy, hCst_occupancy, hCst_bandwidth, hCcst_latency = first_fit(networkServices.state[batch], networkServices.service_length[batch], env)
            # first_fit error(workaround)
            hPlacement, hEnergy, hCst_occupancy, hCst_bandwidth, hCcst_latency = 0., 0., 0., 0., 0.

            hPenalty = agent.lambda_occupancy * hCst_occupancy + agent.lambda_bandwidth * hCst_bandwidth + agent.lambda_latency * hCcst_latency
            hLagrangian = hEnergy + hPenalty

            sPlacement, sSvc_bandwidth, sSvc_net_latency, sSvc_cpu_latency, sEnergy, sOccupancy, sLink_used = \
                solver(networkServices.state[batch], networkServices.service_length[batch], env)

            if sPlacement == None:
                sReward[batch] = 0
            else:
                env.clear()
                env.step(networkServices.service_length[batch], networkServices.state[batch], sPlacement)

                assert sSvc_bandwidth == env.bandwidth
                assert sSvc_net_latency == env.link_latency
                assert sSvc_cpu_latency == env.cpu_latency
                assert sEnergy == env.reward
                assert sOccupancy == list(env.cpu_used)
                assert sLink_used == list(env.link_used)

                sReward[batch] = env.reward

            # Print testing
            print("solver reward: ", sReward[batch])
            print("reward: ", best_reward_t[batch])
            print("cstr_occupancy: ", best_constraint_occupancy_t[batch])
            print("cstr_bw: ", best_constraint_bandwidth_t[batch])
            print("cstr_lat: ", best_constraint_latency_t[batch])
            print("fl_reward: ", fl_reward_t[batch])
            print("fl_occupancy: ", fl_constraint_occupancy_t[batch])
            print("fl_bw: ", fl_constraint_bandwidth_t[batch])
            print("fl_lat: ", fl_constraint_latency_t[batch])

            # Save in test.csv
            csvData = [' batch: {}'.format(batch),
                       ' network_service: {}'.format(networkServices.state[batch]),
                       ' placement: {}'.format(best_placement_t[batch]),
                       ' reward: {}'.format(best_reward_t[batch]),
                       ' penalty: {}'.format(best_penalty_t[batch]),
                       ' solver_placement: {}'.format(sPlacement),
                       ' solver_reward: {}'.format(sReward[batch]),
                       ' fl_placement: {}'.format(fl_placement_t[batch]),
                       ' fl_reward: {}'.format(fl_reward_t[batch]),
                       ' fl_penalty: {}'.format(fl_penalty_t[batch])]

            filePath = '{}_test.csv'.format(config.load_from)
            with open(filePath, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(csvData)

            csvFile.close()

def app(config):

    # session preparations
    env, networkServices, agent, saver = session_preparation()

    print("Starting session ...")

    with tf.Session() as sess:

        # Activate Tensorflow CLI debugger
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Activate Tensorflow debugger in Tensorboard
        #sess = tf_debug.TensorBoardDebugWrapperSession(
        #    sess=sess,
        #    grpc_debug_server_addresses=['localhost:6064'],
        #    send_traceback_and_source_code=True)

        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Print total number of parameters
        print_trainable_parameters()

        # Learn model
        if config.learn_mode:
            learning(sess, config, env, networkServices, agent, saver)

        else:
            inference(sess, config, env, networkServices, agent, saver)

    # session complete and reset graphing
    tf.reset_default_graph()


if __name__ == "__main__":

    config, _ = get_config()
    app(config)

    # path = config.save_to
    #
    # # list all runs and execute
    # for i in range(config.num_runs):
    #     config.save_to = path + str(i)


