#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Dagui Chen
Email: goblin_chen@163.com
Data: 2017-07-04

``````````````````````````````````````
Agent Class
Modified from
    https://github.com/ebonyclock/vizdoom_cig2017/blob/master/intelact/IntelAct_track2/agent/agent.py
"""

__author__ = "Dagui Chen"


import json
import itertools as it
import os
import numpy as np
import layers as ly
import tensorflow as tf
import copy


class Agent(object):
    """Agent Class
    """
    def __init__(self, sess, config_file, simulator):
        """__init__ function
        Args:
        sess: tf.Session()
        config_file: json file
        simulator: A instance of DoomSimulator
        """
        self.sess = sess
        with open(config_file, 'r') as agent_fp:
            agent_config = json.load(agent_fp)

        # test paramter, pulic attribute
        self.history_length = agent_config['history_length']
        self.test_checkpoint = agent_config['test_checkpoint'].encode('ascii')
        self.test_objective_params = tuple(map(np.array, agent_config['test_objective_params']))
        self.verbose = agent_config['verbose']
        if self.verbose:
            print 'test_objective_param:', self.test_objective_params

        # control paramter
        self.__opposite_button_pairs = agent_config['opposite_button_pairs']
        self.__discrete_controls_manual = agent_config['discrete_controls_manual']
        self.__discrete_controls = simulator.discrete_controls
        self.__prepare_controls_and_actions()

        # measurement
        if 'measure_for_net_init' in agent_config:
            self.__measure_for_net = []
            for his_len in range(self.history_length):
                self.__measure_for_net += [i + simulator.num_measure * his_len
                                           for i in agent_config['measure_for_net_init']]
            self.__measure_for_net = np.array(self.__measure_for_net)
        else:
            self.__measure_for_net = np.arange(self.history_length * simulator.num_measure)

        if 'measure_for_manual_init' in agent_config and agent_config['measure_for_manual_init']:
            self.__measure_for_manual = np.array([i + simulator.num_measure * (self.history_length - 1) for
                                                  i in agent_config['measure_for_manual_init']])
        else:
            self.__measure_for_manual = []

        # shape parameter
        self.__state_imgs_shape = (self.history_length * simulator.num_channels,
                                   simulator.resolution[1], simulator.resolution[0])
        self.__state_measure_shape = (len(self.__measure_for_net), )

        # network parameter
        self.__conv_params = agent_config['conv_params']
        self.__fc_img_params = agent_config['fc_img_params']
        self.__fc_measure_params = agent_config['fc_measure_params']
        self.__fc_joint_params = agent_config['fc_joint_params']
        self.__num_future_steps = agent_config['num_future_steps']
        self.__target_dim = self.__num_future_steps * len(agent_config['measure_for_net_init'])

        self.__build_model()

    def get_img_shape(self):
        return self.__state_imgs_shape

    def __prepare_controls_and_actions(self):
        self.__discrete_controls_to_net = np.array([i for i in range(len(self.__discrete_controls))
                                                    if i not in self.__discrete_controls_manual])
        self.__num_manual_controls = len(self.__discrete_controls_manual)
        self.__net_discrete_actions = []
        if not self.__opposite_button_pairs:
            for perm in it.product([False, True], repeat=len(self.__discrete_controls_to_net)):
                self.__net_discrete_actions.append(list(perm))
        else:
            for perm in it.product([False, True], repeat=len(self.__discrete_controls_to_net)):
                act = list(perm)
                valid = True
                for button_p in self.__opposite_button_pairs:
                    if act[button_p[0]] and act[button_p[1]]:
                        valid = False
                if valid:
                    self.__net_discrete_actions.append(act)

        self.__num_net_discrete_actions = len(self.__net_discrete_actions)
        self.__action_to_index = {tuple(val): ind for (ind, val) in enumerate(self.__net_discrete_actions)}
        self.__net_discrete_actions = np.array(self.__net_discrete_actions)
        self.__onehot_discrete_acitons = np.eye(self.__num_net_discrete_actions)

    def __preprocess_input_images(self, rawimage):
        return rawimage / 255. - 0.5

    def __preprocess_input_measure(self, rawmeasure):
        return rawmeasure / 100. - 0.5

    def __preprocess_actions(self, acts):
        to_net_acts = acts[:, self.__discrete_controls_to_net]
        return self.__onehot_discrete_acitons[np.array([self.__action_to_index[tuple(act)]
                                                        for act in to_net_acts.tolist()])]

    def __postprocess_actions(self, acts_net, acts_manual=None):
        out_actions = np.zeros((acts_net.shape[0], len(self.__discrete_controls)), dtype=np.int)
        out_actions[:, self.__discrete_controls_to_net] = self.__net_discrete_actions[acts_net]
        if acts_manual is not None:
            out_actions[:, self.__discrete_controls_manual] = acts_manual
        return out_actions

    def random_actions(self, num_samples):
        acts_net = np.random.randint(0, self.__num_net_discrete_actions, num_samples)
        acts_manual = np.zeros((num_samples, self.__num_manual_controls), dtype=np.bool)
        return self.__postprocess_actions(acts_net, acts_manual)

    def __make_net(self, input_images, input_measure, input_actions, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        fc_val_params = copy.deepcopy(self.__fc_joint_params)
        fc_val_params[-1]['out_dims'] = self.__target_dim

        fc_adv_params = copy.deepcopy(self.__fc_joint_params)
        fc_adv_params[-1]['out_dims'] = len(self.__net_discrete_actions) * self.__target_dim

        if self.verbose:
            print 'fc_val_params:', fc_val_params
            print 'fc_adv_params:', fc_adv_params

        p_img_conv = ly.conv_encoder(input_images, self.__conv_params, 'p_img_conv', msra_coeff=0.9)
        p_img_fc = ly.fc_net(ly.flatten(p_img_conv), self.__fc_img_params, 'p_img_fc', msra_coeff=0.9)
        p_meas_fc = ly.fc_net(input_measure, self.__fc_measure_params, 'p_meas_fc', msra_coeff=0.9)
        p_val_fc = ly.fc_net(tf.concat([p_img_fc, p_meas_fc], 1),
                             fc_val_params, 'p_val_fc', last_linear=True, msra_coeff=0.9)
        p_adv_fc = ly.fc_net(tf.concat([p_img_fc, p_meas_fc], 1),
                             fc_adv_params, 'p_adv_fc', last_linear=True, msra_coeff=0.9)
        p_adv_fc_nomean = p_adv_fc - tf.reduce_mean(p_adv_fc, reduction_indices=1, keep_dims=True)

        self.__pred_all_nomean = tf.reshape(p_adv_fc_nomean, [-1, len(self.__net_discrete_actions), self.__target_dim])
        self.__pred_all = self.__pred_all_nomean + tf.reshape(p_val_fc, [-1, 1, self.__target_dim])
        self.__pred_relevant = tf.boolean_mask(self.__pred_all, tf.cast(input_actions, tf.bool))

    def __build_model(self):
        self.__input_images = tf.placeholder(
            tf.float32,
            [None, self.__state_imgs_shape[1], self.__state_imgs_shape[2], self.__state_imgs_shape[0]],
            name='input_images')
        self.__input_measure = tf.placeholder(
            tf.float32,
            [None] + list(self.__state_measure_shape), name='input_measurements')
        self.__input_actions = tf.placeholder(tf.float32, [None, self.__num_net_discrete_actions], name="input_actions")

        self.__input_images_preprocessed = self.__preprocess_input_images(self.__input_images)
        self.__input_measure_preprocessed = self.__preprocess_input_measure(self.__input_measure)

        self.__make_net(self.__input_images_preprocessed, self.__input_measure_preprocessed, self.__input_actions)
        self.__saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def __act_net(self, state_imgs, state_meas, objective):
        predictions = self.sess.run(self.__pred_all, feed_dict={self.__input_images: state_imgs,
                                    self.__input_measure: state_meas[:, self.__measure_for_net]})
        objectives = np.sum(predictions[:, :, objective[0]] * objective[1][None, None, :], axis=2)
        curr_action = np.argmax(objectives, axis=1)
        return curr_action

    def __act_manual(self, state_meas):
        if len(self.__measure_for_manual):
            # [AMMO2, AMMO3, AMMO4, AMMO5, AMMO6, AMMO7, WEAPON2,
            # WEAPON3 WEAPON4 WEAPON5 WEAPON6 WEAPON7 SELECTED_WEAPON]
            assert len(self.__measure_for_manual) == 13
            # [SELECT_WEAPON2 SELECT_WEAPON3 SELECT_WEAPON4 SELECT_WEAPON5 SELECT_WEAPON6 SELECT_WEAPON7]
            curr_action = np.zeros((state_meas.shape[0], self.__num_manual_controls), dtype=np.int)
            for ns in range(state_meas.shape[0]):
                curr_ammo = state_meas[ns, self.__measure_for_manual[:6]]
                curr_weapons = state_meas[ns, self.__measure_for_manual[6:12]]
                if self.verbose:
                    print 'current ammo:', curr_ammo
                    print 'current weapons:', curr_weapons
                available_weapons = np.logical_and(curr_ammo >= np.array([1, 2, 1, 1, 1, 40]), curr_weapons)
                if any(available_weapons):
                    best_weapon = np.nonzero(available_weapons)[0][-1]
                    if not state_meas[ns, self.__measure_for_manual[12]] == best_weapon + 2:
                        curr_action[ns, best_weapon] = 1
            return curr_action
        else:
            return []

    def act(self, state_imgs, state_meas, objective):
        """Act function
        Args:
        state_imgs: images
        state_meas: measures
        objective: objective
        """
        return self.__postprocess_actions(
            self.__act_net(state_imgs, state_meas, objective),
            self.__act_manual(state_meas)), None

    def load(self, checkpoint_dir):
        """Load checkpoint
        """
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.__saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
