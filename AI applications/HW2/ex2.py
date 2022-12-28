ids = ["212778229", "325069565"]

import networkx as nx
import logging
from copy import deepcopy
from collections import defaultdict
from utils import FIFOQueue
import itertools
import random
import numpy as np
from tqdm import tqdm
from time import time

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1

# random.seed(10)


TAXIS = 0
# Taxi params
NAME = 0
LOC = 1
FUEL = 2
CAP = 3

PASSENGERS = 1
# Passenger params
DESTINATION = 2
POSSIBLE_DESTINATIONS = 3
PROBABILITY = 4

TURNS_TO_GO = 2


class OptimalTaxiAgent:
    def __init__(self, initial):
        self.map = initial["map"]
        self.initial = self.init_to_tup(initial)
        self.tName2id = self.taxi_name_to_id()
        self.pName2id = self.passenger_name_to_id()
        # self.add_change_prob(self.initial)
        self.graph = self.build_graph()
        self.state = self.initial
        self.all_actions_dict = dict()
        self.next_dict = dict()
        self.inner_prob_dict = dict()
        # self.policy = self.policy_iterations(max_iterations=self.initial[TURNS_TO_GO])

        self.policy = self.value_iterations()
        # print(f"the expected value is {self.policy[self.initial]}")

    def init_to_tup(self, initial):
        """converts the initial state to a tuple"""
        taxis = initial["taxis"]
        passengers = initial["passengers"]
        passengers_tups = tuple([(passenger, passengers[passenger]["location"],
                                  passengers[passenger]["destination"], passengers[passenger]['possible_goals'],
                                  ((len(passengers[passenger]['possible_goals']) - 1) / (
                                      len(passengers[passenger]['possible_goals'])) *
                                   passengers[passenger]['prob_change_goal'])) for passenger in passengers])
        taxis_tups = tuple([(taxi, taxis[taxi]["location"], taxis[taxi]["fuel"], taxis[taxi]["capacity"])
                            for taxi in taxis])
        initial = (taxis_tups, passengers_tups, initial["turns to go"])
        return initial

    def taxi_name_to_id(self):
        d = {}
        idx = 0
        for taxi in self.initial[TAXIS]:
            d[taxi[NAME]] = idx
        return d

    def passenger_name_to_id(self):
        d = {}
        idx = 0
        for passenger in self.initial[PASSENGERS]:
            d[passenger[NAME]] = idx
        return d

    # def add_change_prob(self, initial_state):
    #     """add the probability of change to the initial state"""
    #     for passenger in initial_state[PASSENGERS]:
    #         passenger[PROBABILITY] = 1 / (len(passenger[POSSIBLE_DESTINATIONS]) - 1) * passenger[PROBABILITY]

    def next(self, state, action):
        """runs the given action form the given state and returns the new state"""
        return self.apply(state, action)

    def apply(self, state, action):
        """
        apply the action to the state
        """
        if action[0] == "reset":
            return self.initial[0], self.initial[1], state[TURNS_TO_GO] - 1
        state = (state[TAXIS], state[PASSENGERS], state[TURNS_TO_GO] - 1)
        if action[0] == "terminate":
            return None
        for atomic_action in action:
            next = self.apply_atomic_action(state, atomic_action)
        return next

    def state_to_tup(self, state):
        """converts a state to a tuple"""
        return tuple([tuple(tup) for tup in state[TAXIS]]), tuple([tuple(tup) for tup in state[PASSENGERS]]), state[
            TURNS_TO_GO]

    def apply_atomic_action(self, state, atomic_action):
        """
        apply an atomic action to the state
        """
        if atomic_action[0] == 'wait':
            return state
        old_state = state
        state = [[list(tup) for tup in old_state[TAXIS]], [list(tup) for tup in old_state[PASSENGERS]],
                 old_state[TURNS_TO_GO]]
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            state[TAXIS][self.tName2id[taxi_name]][LOC] = atomic_action[2]
            state[TAXIS][self.tName2id[taxi_name]][FUEL] -= 1
            return self.state_to_tup(state)
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            state[TAXIS][self.tName2id[taxi_name]][CAP] -= 1
            state[PASSENGERS][self.pName2id[passenger_name]][LOC] = taxi_name
            return self.state_to_tup(state)
        elif atomic_action[0] == 'drop off':
            passenger_name = atomic_action[2]
            state[PASSENGERS][self.pName2id[passenger_name]][LOC] = state[TAXIS][self.tName2id[taxi_name]][LOC]
            state[TAXIS][self.tName2id[taxi_name]][CAP] += 1
            return self.state_to_tup(state)
        elif atomic_action[0] == 'refuel':
            state[TAXIS][self.tName2id[taxi_name]][FUEL] = self.initial[TAXIS][self.tName2id[taxi_name]][FUEL]
            return self.state_to_tup(state)
        # else:
        #     raise NotImplemented

    def reward(self, state, action):
        """
        return the reward of performing the action in the state
        """
        if action == ('reset',):
            return -RESET_PENALTY
        if action == 'terminate':
            return 0

        reward = 0
        for atomic_action in action:
            if atomic_action == 'refuel':
                reward -= REFUEL_PENALTY
            if atomic_action[0] == 'drop off':
                # passenger_name = atomic_action[2]
                # if state[PASSENGERS][self.pName2id[passenger_name]][LOC] == \
                #         state[PASSENGERS][self.pName2id[passenger_name]][DESTINATION]:
                reward += DROP_IN_DESTINATION_REWARD
        return reward

    def act(self, state):
        """
        return the action to perform in the state
        """
        state = self.init_to_tup(state)
        action = self.policy[state]
        return action if action != ("reset",) else "reset"

    def all_actions(self, state):
        if state not in self.all_actions_dict:
            self.all_actions_dict[state] = self.all_actions_aux(state)
        return self.all_actions_dict[state]

    def all_actions_aux(self, state):
        """
        return all possible actions
        """
        all_actions = []
        taxi_actions = {}
        for taxi in state[TAXIS]:
            taxi_actions[taxi[NAME]] = []
            taxi_name = taxi[NAME]
            # move actions
            if taxi[FUEL] > 0:
                for neighbor in self.graph.neighbors(taxi[LOC]):
                    taxi_actions[taxi_name].append(('move', taxi_name, neighbor))
            # pick up actions
            for passenger in state[PASSENGERS]:
                # passenger = state[PASSENGERS][self.pName2id[passenger_name]]
                # passenger_name = passenger_name[NAME]
                if passenger[LOC] == taxi[LOC] and taxi[CAP] > 0 and passenger[DESTINATION] != passenger[LOC]:
                    taxi_actions[taxi_name].append(('pick up', taxi_name, passenger[NAME]))
                # drop off actions
                if passenger[LOC] == taxi_name and passenger[DESTINATION] == taxi[LOC]:
                    taxi_actions[taxi_name].append(('drop off', taxi_name, passenger[NAME]))
            # refuel actions
            i, j = state[TAXIS][self.tName2id[taxi_name]][LOC]
            if self.map[i][j] == 'G':
                taxi_actions[taxi_name].append(('refuel', taxi_name))
            # wait actions
            taxi_actions[taxi_name].append(('wait', taxi_name))
        # reset action
        all_actions = list(itertools.product(*taxi_actions.values()))
        if len(state[TAXIS]) > 1:
            for action in all_actions:
                for taxi_action in action:
                    for taxi2_action in action:
                        if taxi_action[1] != taxi2_action[1]:
                            if (taxi_action[0] == 'move' and taxi2_action[0] == 'move' and taxi_action[2] == taxi2_action[
                                2]) or \
                                    (taxi_action[0] == 'move' and taxi2_action[0] != 'move' and taxi_action[2] ==
                                     state[TAXIS][self.tName2id[taxi2_action[1]]][LOC]):
                                all_actions.remove(action)
        all_actions.append(('reset',))
        # terminate action
        # all_actions.append('terminate')
        return all_actions

    def generate_all_states(self):
        """uses all_actions to generate all possible states, kinda runs BFS"""
        turns_to_go = self.state[TURNS_TO_GO] + 1
        all_states = defaultdict(lambda: set())
        all_states[0] = set((self.state,))
        for i in (tq := tqdm(range(1, turns_to_go + 1), leave=False)):
            # diff = new_states.difference(old_states)
            # old_states = new_states.copy()
            count = 0
            for state in all_states[i - 1]:
                self.all_actions_dict[state] = {}
                self.next_dict[state] = {}
                count += 1
                for action in self.all_actions_aux(state):
                    new_state = self.next(state, action)
                    self.next_dict[state][action] = new_state
                    all_new_states = self.split_across_MDP(new_state)
                    all_states[i].update(all_new_states)
                    if action in self.all_actions_dict[state]:
                        self.all_actions_dict[state][action].append(list(all_new_states))
                    else:
                        self.all_actions_dict[state][action] = list(all_new_states)
            tq.set_description(f"Generating all states")
        return all_states

    def split_across_MDP(self, state):
        """
        split the state across MDPs
        """
        new_states = set()
        if len(state[PASSENGERS]) > 1:
            possible_goals = [state[PASSENGERS][self.pName2id[passenger_name]][POSSIBLE_DESTINATIONS] for passenger_name
                              in
                              [passenger[NAME] for passenger in state[PASSENGERS]]]

            for goals in itertools.product(*possible_goals):
                pass_list = list(list(passenger) for passenger in state[PASSENGERS])
                for i, goal in enumerate(goals):
                    pass_list[i][DESTINATION] = goal
                new_state = (state[TAXIS], tuple(tuple(passenger) for passenger in pass_list), state[TURNS_TO_GO])
                # _state = deepcopy(state)
                # for goal, passenger in zip(goals, _state[PASSENGERS]):
                #     passenger[DESTINATION] = goal
                new_states.add(new_state)
        else:
            passenger = state[PASSENGERS][0]
            for goal in passenger[POSSIBLE_DESTINATIONS]:
                new_state = (state[TAXIS], (
                    (passenger[NAME], passenger[LOC], goal, passenger[POSSIBLE_DESTINATIONS], passenger[PROBABILITY]),),
                             state[TURNS_TO_GO])
                new_states.add(new_state)
        return new_states

    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.map), len(self.map[0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.map[node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g

    def inner(self, state, action, new_state, old_values):
        """returns the probability of the new state given the state and action"""
        if (state, action, new_state) in self.inner_prob_dict:
            return self.inner_prob_dict[(state, action, new_state)] * old_values[new_state]
        else:
            prob = 1
            # next_state = self.next(state, action)
            next_state = self.next_dict[state][action]
            for curr_passenger, new_passenger in zip(next_state[PASSENGERS], new_state[PASSENGERS]):
                if curr_passenger[DESTINATION] != new_passenger[DESTINATION]:
                    prob *= curr_passenger[PROBABILITY]
                else:
                    prob *= (1 - curr_passenger[PROBABILITY])
            self.inner_prob_dict[(state, action, new_state)] = prob
            return self.inner_prob_dict[(state, action, new_state)] * old_values[new_state]

    def expected_value(self, state, action, old_values):
        """
        return the expected value of the action in the state
        """
        if state[TURNS_TO_GO] == 0:
            return 0
        next_states = self.all_actions_dict[state][action]
        reward = self.reward(state, action)
        ex = sum([self.inner(state, action, next_state, old_values) for next_state in next_states])
        return reward + ex

    def policy_iterations(self, max_iterations=200, gamma=0.9, epsilon=1e-21):
        """
        policy iteration algorithm
        """
        # initialize value function
        all_state_list = self.generate_all_states()
        best_action = {}
        num = sum([len(list(all_state_list[i])) for i in range(max_iterations + 1)])
        # values = dict([(s, 0) for s in all_state_list])
        # values = defaultdict(lambda: 0)
        values = dict([(s, 0) for i in range(max_iterations + 1) for s in all_state_list[i]])
        # initialize policy
        policy = dict(
            [(s, random.choice(self.all_actions(s))) for i in range(max_iterations + 1) for s in all_state_list[i]])
        # initialize iteration counter
        counter = 0
        # initialize delta
        delta = float('inf')
        for iter in (tq := tqdm(range(max_iterations), leave=True, position=0)):
            if delta <= epsilon:
                print("-------------- broke --------------")
                break
            delta = 0
            old_values = values.copy()
            for i in range(max_iterations + 1):
                for state in all_state_list[i]:
                    old_value = old_values[state]
                    counter += 1
                    for action in self.all_actions_dict[state].keys():
                        expected = self.expected_value(state, action, gamma, old_values)
                        if expected > values[state]:
                            values[state] = expected
                            policy[state] = action

                    values[state] = self.expected_value(state, action, gamma, old_values)
                    delta = max(delta, abs(old_value - values[state]))
            tq.set_description(f"Value Iterations")

        return policy

    def value_iterations(self, max_iterations=100):
        """
        value iteration algorithm, using dynamic programming
        """
        all_state_list = self.generate_all_states()
        # print(sum([len(list(all_state_list[i])) for i in range(max_iterations + 1)]))
        values = dict([(s, 0) for i in range(max_iterations + 1) for s in all_state_list[i]])
        policy = dict(
            [(s, random.choice(list(self.all_actions_dict[s].keys()))) for i in range(max_iterations + 1) for s in
             all_state_list[i]])
        # for iter in (tq := tqdm(range(max_iterations), leave=True, position=0)):
        start = time()
        for i in range(max_iterations - 1, -1, -1):
            for state in all_state_list[i]:
                actions = list(self.all_actions_dict[state].keys())
                action_values = [self.expected_value(state, a, values) for a in actions]
                values[state] = max(action_values)
                policy[state] = actions[np.argmax(action_values)]
        end = time()
        print("Time taken: ", end - start)
        print(f"{values[list(all_state_list[0])[0]]=}")
        return policy


class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented
