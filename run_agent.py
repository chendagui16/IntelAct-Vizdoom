#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Dagui Chen
Email: goblin_chen@163.com
Data: 2017-07-05

``````````````````````````````````````
Run Agent
Modified by
    https://github.com/ebonyclock/vizdoom_cig2017/blob/master/intelact/IntelAct_track2/run_agent.py
"""

__author__ = "Dagui Chen"


import numpy as np
import tensorflow as tf
from Simulator import DoomSimulator
from agent import Agent
from time import sleep


def main():
    """Main function
    Test the checkpoint
    """
    simulator_config = 'config/simulator.json'
    print 'Starting simulator...'
    simulator = DoomSimulator(simulator_config)
    simulator.add_bots(10)
    print 'Simulator started!'

    agent_config = 'config/agent.json'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    print 'Creating agent...'
    ag = Agent(sess, agent_config, simulator)
    print 'load model...'
    loadstatus = ag.load('./checkpoints')
    if not loadstatus:
        raise IOError

    img_buffer = np.zeros((ag.history_length, simulator.num_channels, simulator.resolution[1], simulator.resolution[0]))
    measure_buffer = np.zeros((ag.history_length, simulator.num_measure))
    curr_step = 0
    term = False
    acts_to_replace = [a + b + c + d for a in [[0, 0], [1, 1]]
                       for b in [[0, 0], [1, 1]] for c in [[0]] for d in [[0], [1]]]
    replacement_act = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # Action0-5: MOVE_FORWARD MOVE_BACKWARD TURN_LEFT TURN_RIGHT ATTACK SPPED
    # Action6-11: SELECT_WEAPON2 ~ SELECT_WEAPON7

    while not term:
        if curr_step < ag.history_length:
            img, meas, reward, term = simulator.step(np.squeeze(ag.random_actions(1)).tolist())
        else:
            state_imgs = img_buffer[np.arange(curr_step - ag.history_length, curr_step) % ag.history_length]
            state_imgs = np.reshape(state_imgs, (1,) + ag.get_img_shape())
            state_imgs = np.transpose(state_imgs, [0, 2, 3, 1])
            state_meas = measure_buffer[np.arange(curr_step - ag.history_length, curr_step) % ag.history_length]
            state_meas = np.reshape(state_meas, (1, ag.history_length * simulator.num_measure))
            curr_act = np.squeeze(ag.act(state_imgs, state_meas, ag.test_objective_params)[0]).tolist()
            if curr_act[:6] in acts_to_replace:
                curr_act = replacement_act
            img, meas, reward, term = simulator.step(curr_act)
            if (not (meas is None)) and meas[0] > 30.:
                meas[0] = 30.
        simulator.show_info()
        sleep(0.02)
        if not term:
            img_buffer[curr_step % ag.history_length] = img
            measure_buffer[curr_step % ag.history_length] = meas
            curr_step += 1
    simulator.close_game()


if __name__ == "__main__":
    main()
