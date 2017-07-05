#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Dagui Chen
Email: goblin_chen@163.com
Data: 2017-06-29

``````````````````````````````````````
Simulator Class
Modified by
    https://github.com/ebonyclock/vizdoom_cig2017/blob/master/intelact/IntelAct_track2/agent/doom_simulator.py
"""

__author__ = "Dagui Chen"
import re
import random
import json
import vizdoom
import numpy as np


class DoomSimulator(object):
    """Simulator Class
    """
    def __init__(self, config_fp):
        with open(config_fp, 'r') as config_file:
            config = json.load(config_file)
        self.game_config = config['game_config'].encode('ascii')
        self.frame_skip = config['frame_skip']
        self.game_args = config['game_args'].encode('ascii')
        self._game = vizdoom.DoomGame()
        setstatus = self._game.load_config(self.game_config)
        if not setstatus:
            raise IOError
        self._game.add_game_args(self.game_args)

        # Parse the game_config file
        with open(self.game_config, 'r') as config_file:
            game_config = config_file.read()
        # set resolution
        match_string = r'screen_resolution[\s]*\=[\s]*RES_(\d*)X(\d*)\s'
        match = re.search(match_string, game_config)
        self.resolution = (int(match.group(1)), int(match.group(2)))

        match = re.search(r'screen_format[\s]*\=[\s]*([^\s]*)\s', game_config)
        self.color_mode = match.group(1).strip()
        if self.color_mode == 'GRAY8':
            self.num_channels = 1
        elif self.color_mode == 'CRCGCB':
            self.num_channels = 3
        else:
            raise ValueError

        match_string = r'available_buttons[\s]*\=[\s]*\{([^\}]*)\}'
        match = re.search(match_string, game_config)
        self.avail_controls = match.group(1).split()
        cont_controls = np.array(
            [bool(re.match('.*_DELTA', c)) for c in self.avail_controls])
        discr_controls = np.invert(cont_controls)
        self.continuous_controls = np.squeeze(np.nonzero(cont_controls))
        self.discrete_controls = np.squeeze(np.nonzero(discr_controls))
        self.num_buttons = self._game.get_available_buttons_size()
        assert self.num_buttons == len(self.discrete_controls) \
            + len(self.continuous_controls)
        assert not self.continuous_controls
        self.num_measure = self._game.get_available_game_variables_size()
        self._game.init()

    def close_game(self):
        """Close the game
        """
        self._game.close()

    def step(self, action):
        """Return img, measure, reward, term
        """

        reward = self._game.make_action(action, self.frame_skip)
        if self._game.is_episode_finished():
            return None, None, reward, True

        state = self._game.get_state()
        img = state.screen_buffer
        measure = state.game_variables
        term = self._game.is_episode_finished()
        return img, measure, reward, term

    def get_random_action(self):
        """Get random action
        """
        return [(random.random() >= .5) for _ in range(self.num_buttons)]

    def is_new_episode(self):
        """identify whether is new episode
        """
        return self._game.is_new_episode()

    def restart_game(self):
        """restart game
        """
        self._game.respawn_player()

    def add_bots(self, numbots):
        for _ in range(numbots):
            self._game.send_game_command("addbot")

    def show_info(self):
        s = self._game.get_state()
        if s.number % 175 == 0:
            print "Frags", self._game.get_game_variable(vizdoom.GameVariable.FRAGCOUNT)


def main():
    """Test the Simulator
    """
    from time import sleep
    simulator = DoomSimulator('config/simulator.json')
    for _ in range(10000):
        action = simulator.get_random_action()
        _, _, _, term = simulator.step(action)
        if term:
            simulator.restart_game()
        sleep(0.01)
    simulator.close_game()


if __name__ == "__main__":
    main()
