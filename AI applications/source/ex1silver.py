import search
import random
import math
import json
from scipy.spatial.distance import euclidean
import itertools
from itertools import combinations
import time
import numpy as np
import networkx as nx
from collections import OrderedDict
import pandas as pd
import hashlib

ids = ["212672034", "212724462"]
ACTIONS = ['move', 'pick up', 'drop off', 'refuel', 'wait']


def d2t(d):
    if type(d) != dict:
        return d
    lst = []
    for k in sorted(d.keys()):
        lst.append((k, d2t(d[k])))
    return tuple(lst)


def t2d(t):
    if type(t) != tuple:
        return t
    if type(t[0]) != str and type(t[0]) != tuple:
        return t

    d = {}
    for item in t:
        if item[1] != ():
            d[item[0]] = t2d(item[1])
    return d


class HashableDict(dict):
    def __hash__(self):
        return hash(d2t(self.__dict__))


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""

        search.Problem.__init__(self, initial)
        self.taxis = list(initial['taxis'].keys())
        self.passengers = list(initial['passengers'].keys())

        for p in self.passengers:
            initial['passengers'][p]['in_taxi'] = 'null'
        for t in self.taxis:
            initial['taxis'][t]['max_fuel'] = initial['taxis'][t]['fuel']
            initial['taxis'][t]['max_capacity'] = initial['taxis'][t]['capacity']

        init_state_str = json.dumps({'taxis': initial['taxis'], 'passengers': initial['passengers']})

        # print({'taxis': initial['taxis'], 'passengers': initial['passengers']})
        # print(t2d(d2t({'taxis': initial['taxis'], 'passengers': initial['passengers']})))

        self.initial = init_state_str
        self.map = initial['map']

        self.map_w = len(self.map[0])
        self.map_h = len(self.map)

        G = nx.Graph()


        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                if (self.map[y][x] != 'I'):
                    G.add_node((y, x))


        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                if G.has_node((y, x)):
                    if G.has_node((y + 1, x)):
                        G.add_edge((y, x), (y + 1, x))

                    if G.has_node((y - 1, x)):
                        G.add_edge((y, x), (y - 1, x))

                    if G.has_node((y, x + 1)):
                        G.add_edge((y, x), (y, x + 1))

                    if G.has_node((y, x - 1)):
                        G.add_edge((y, x), (y, x - 1))

        self.length = dict(nx.all_pairs_shortest_path_length(G))

        G1 = nx.Graph()
        G1.add_node((-1, -1))

        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                if (self.map[y][x] != 'I'):
                    G1.add_node((y, x))
                    if (self.map[y][x] == 'G'):
                        G1.add_edge((y, x), (-1, -1))

        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                if G1.has_node((y, x)):
                    if G1.has_node((y + 1, x)):
                        G1.add_edge((y, x), (y + 1, x))

                    if G1.has_node((y - 1, x)):
                        G1.add_edge((y, x), (y - 1, x))

                    if G1.has_node((y, x + 1)):
                        G1.add_edge((y, x), (y, x + 1))

                    if G1.has_node((y, x - 1)):
                        G1.add_edge((y, x), (y, x - 1))

        self.length_gas = dict(nx.all_pairs_shortest_path_length(G1))

        self.G = G

        self.min_heuristic = np.inf


    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        if type(state) == dict:
            state_dict = state.copy()
        else:
            state_dict = json.loads(state)


            # print(f'STARTED LOOKING FOR ACTION FROM {state_dict}')
        taxi_actions = {}
        passengers = list(state_dict['passengers'].keys())
        taxis = list(state_dict['taxis'].keys())

        for t in taxis:
            taxi_actions[t] = [('wait', t)]
            t_data = state_dict['taxis'][t]
            curr_loc = t_data['location']

            # refuel
            y, x = curr_loc
            if self.map[y][x] == 'G':
                taxi_actions[t].append(('refuel', t))

            # move
            # if t_data['fuel'] > 0:
            #     if y - 1 >= 0:
            #         if self.map[y - 1][x] != "I":
            #             taxi_actions[t].append(('move', t, (y - 1, x)))

            #     if y + 1 < self.map_h:
            #         if self.map[y + 1][x] != "I":
            #             taxi_actions[t].append(('move', t, (y + 1, x)))

            #     if x - 1 >= 0:
            #         if self.map[y][x - 1] != "I":
            #             taxi_actions[t].append(('move', t, (y, x - 1)))

            #     if x + 1 < self.map_w:
            #         if self.map[y][x + 1] != "I":
            #             taxi_actions[t].append(('move', t, (y, x + 1)))

            if t_data['fuel'] > 0:
                for loc in self.G.neighbors(tuple(curr_loc)):
                    y1, x1 = loc
                    taxi_actions[t].append(('move', t, (y1, x1)))

            # Pick up
            available_seats = t_data['capacity']
            if available_seats > 0:
                for person in passengers:
                    p_data = state_dict['passengers'][person]
                    if p_data['location'] == curr_loc and p_data['in_taxi'] == 'null':
                        taxi_actions[t].append(('pick up', t, person))
        # Drop off
        for p in passengers:
            t = state_dict['passengers'][p]['in_taxi']
            if t != 'null':
                if state_dict['passengers'][p]['destination'] == state_dict['taxis'][t]['location']:
                    taxi_actions[t].append(('drop off', t, p))

        if len(taxis) <= 1:
            output = list(itertools.product(*taxi_actions.values()))

        else:
            def not_crashing(action_prod):
                # print(action_prod)
                final_locs = [tuple(a[2]) if a[0] == 'move' else tuple(state_dict['taxis'][a[1]]['location']) for a in
                              action_prod]
                # print(final_locs)
                return len(set(final_locs)) == len(final_locs)
            output =list(filter(not_crashing, itertools.product(*taxi_actions.values())))
        print(len(output))
        return output

    def result(self, state, actions):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        if type(state) == dict:
            state_dict = state.copy()
        else:
            state_dict = json.loads(state)

        if len(actions) == 0:
            return state_dict

        if type(actions[0]) != tuple:  # means there is only one taxi and one action
            actions = [actions]

        taxis = list(state_dict['taxis'].keys())

        for action in actions:

            action_name = action[0]
            if action_name == 'wait':
                continue

            # Move
            if action_name == 'move':
                state_dict['taxis'][action[1]]['location'] = action[2]
                state_dict['taxis'][action[1]]['fuel'] -= 1

            # Pick up
            elif action_name == 'pick up':
                state_dict['taxis'][action[1]]['capacity'] -= 1
                state_dict['passengers'][action[2]]['in_taxi'] = action[1]

            # Drop off
            elif action_name == 'drop off':
                del state_dict['passengers'][action[2]]
                state_dict['taxis'][action[1]]['capacity'] += 1


            # Refuel
            elif action_name == 'refuel':
                state_dict['taxis'][action[1]]['fuel'] = state_dict['taxis'][action[1]]['max_fuel']

        return json.dumps(state_dict)

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        if type(state) == dict:
            state_dict = state
        else:
            state_dict = json.loads(state)

        return len(state_dict['passengers']) == 0

    def h_1(self, node: search.Node):
        """
        This is a simple heuristic
        """
        state_dict = json.loads(node.state)

        if len(state_dict['passengers']) == 0:
            return 0

        n_unpicked = 0
        n_picked_not_delivered = 0
        for p in state_dict['passengers']:
            p_loc = state_dict['passengers'][p]['location']
            p_dest = state_dict['passengers'][p]['destination']
            p_taxi = state_dict['passengers'][p]['in_taxi']

            if p_loc != p_dest:
                if p_taxi == 'null':
                    n_unpicked += 1
                else:
                    n_picked_not_delivered += 1

        return (n_unpicked * 2 + n_picked_not_delivered) / len(state_dict['taxis'])

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        state_dict = json.loads(node.state)
        sum_unpicked_dist = 0
        sum_picked_not_delivered = 0

        if len(state_dict['passengers']) == 0:
            return 0

        for p in state_dict['passengers']:
            p_loc = state_dict['passengers'][p]['location']
            p_dest = state_dict['passengers'][p]['destination']
            p_taxi = state_dict['passengers'][p]['in_taxi']

            if p_loc != p_dest:
                if p_taxi == 'null':
                    sum_unpicked_dist += euclidean(p_loc, p_dest)
                else:
                    sum_picked_not_delivered += euclidean(p_dest, state_dict['taxis'][p_taxi]['location'])

        return (sum_unpicked_dist + sum_picked_not_delivered) / len(state_dict['taxis'])

    def h_2_half(self, node):
        def real_dist(a, b):
            if tuple(a) in self.length:
                if tuple(b) in self.length[tuple(a)]:
                    return self.length[tuple(a)][tuple(b)]
            return np.inf

        state_dict = json.loads(node.state)
        dist = [0]

        if len(state_dict['passengers']) == 0:
            return 0

        for p in state_dict['passengers']:
            p_loc = state_dict['passengers'][p]['location']
            p_dest = state_dict['passengers'][p]['destination']
            p_taxi = state_dict['passengers'][p]['in_taxi']

            if p_loc != p_dest:
                if p_taxi == 'null':
                    dist.append(real_dist(p_loc, p_dest))
                else:
                    dist.append(real_dist(p_dest, state_dict['taxis'][p_taxi]['location']))

        return max(dist)

    def manhattan(self, a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b))

    def real_dist(self, a, b):
        return self.manhattan(a, b)
        if tuple(a) in self.length:
            if tuple(b) in self.length[tuple(a)]:
                return self.length[tuple(a)][tuple(b)]
        return np.inf

    def h(self, node):
        """
        This is a *OUR* heuristic
        # """
        return 0
        state_dict = json.loads(node.state)

        if len(state_dict['passengers']) == 0:
            return 0


        dist_lst = np.zeros(len(state_dict['passengers']))
        poss = 0

        for p in state_dict['passengers']:
            p_data = state_dict['passengers'][p]
            if p_data['in_taxi'] != 'null':
                carrying_taxi = p_data['in_taxi']
                dist = self.real_dist(state_dict['taxis'][carrying_taxi]['location'], p_data['destination'])
                if dist <= state_dict['taxis'][carrying_taxi]['fuel']:
                    dist_lst[poss] = dist + 1
                    poss += 1
                else:
                    min_gas_station_dist = self.length_gas[tuple(state_dict['taxis'][carrying_taxi]['location'])][(-1, -1)] - 1
                    if state_dict['taxis'][carrying_taxi]['fuel'] < min_gas_station_dist:
                        return np.inf
                    dist_lst[poss] = dist + 1 + 1
                    poss += 1
            else:
                dist_to_dest = self.real_dist(p_data['location'], p_data['destination'])
                if dist_to_dest == np.inf:
                    return np.inf

                min_oa_dist = np.inf


                for t in list(state_dict['taxis']):
                    if state_dict['taxis'][t]['capacity'] > 0:
                        p_t_dist = self.real_dist(state_dict['taxis'][t]['location'], p_data['location'])
                        min_gas_station_dist = self.length_gas[tuple(state_dict['taxis'][t]['location'])][(-1, -1)] - 1
                        if min_gas_station_dist > state_dict['taxis'][t]['fuel']:
                            min_gas_station_dist = np.inf

                        if p_t_dist + dist_to_dest < state_dict['taxis'][t]['fuel'] and p_t_dist + dist_to_dest < min_oa_dist:
                            min_oa_dist = p_t_dist + dist_to_dest
                        elif max(p_t_dist + dist_to_dest + 1, min_gas_station_dist+1) < min_oa_dist:
                            min_oa_dist = max(p_t_dist + dist_to_dest + 1, min_gas_station_dist + 1)

                if min_oa_dist == np.inf:
                    for p2 in state_dict['passengers']:
                        p2_data = state_dict['passengers'][p2]
                        if p2_data['in_taxi'] != 'null':
                            carrying_taxi = p2_data['in_taxi']
                            if state_dict['taxis'][carrying_taxi]["fuel"] != 0:
                                p2pd = self.real_dist(p_data['location'], p2_data['destination'])
                                t2pd = self.real_dist(p2_data['destination'],
                                                      state_dict['taxis'][carrying_taxi]['location']) + 1
                                if state_dict['taxis'][carrying_taxi]["fuel"] < p2pd + dist_to_dest + t2pd:
                                    p2pd += 1
                                if t2pd + p2pd + dist_to_dest < min_oa_dist:
                                    min_oa_dist = t2pd + p2pd + dist_to_dest

                dist_lst[poss] = min_oa_dist + 2
                poss += 1

            if dist_lst[poss - 1] > self.min_heuristic + 5 and False:
                return dist_lst[poss - 1]

        heuristic = max(dist_lst)  # (sum(dist_lst)/len(dist_lst) + max(dist_lst))/2 + self.h_1(node)

        if heuristic < self.min_heuristic:
            self.min_heuristic = heuristic

        return heuristic
        # avg([min_dist(passenger -> [taxi with enough fuel]) + dist(passenger, dest)] for passenger in taxis]) + h_1


    #def h(self, node: search.Node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""

    #    return self.h_3(node) #self.h_1(node)# + self.h_2_half(node)


def create_taxi_problem(game):
    return TaxiProblem(game)


dicti = {
    'map': [['P', 'P', 'P', 'P', 'P'],
            ['P', 'I', 'P', 'G', 'P'],
            ['P', 'P', 'I', 'P', 'P'],
            ['P', 'P', 'P', 'I', 'P']],
    'taxis': {'taxi 1': {'location': (2, 0), 'fuel': 5, 'capacity': 2},
              'taxi 2': {'location': (0, 1), 'fuel': 6, 'capacity': 2}},
    'passengers': {'Iris': {'location': (0, 0), 'destination': (1, 4)},
                   'Daniel': {'location': (3, 1), 'destination': (2, 1)},
                   'Freyja': {'location': (2, 3), 'destination': (2, 4)},
                   'Tamar': {'location': (3, 0), 'destination': (3, 2)}},
}

'''
actions = [(('wait', 'taxi 1'), ('move', 'taxi 2', (0, 0))), (('wait', 'taxi 1'), ('pick up', 'taxi 2', 'Iris')), (('wait', 'taxi 1'), ('move', 'taxi 2', (0, 1))), (('wait', 'taxi 1'), ('move', 'taxi 2', (0, 2))), (('move', 'taxi 1', (3, 0)), ('move', 'taxi 2', (0, 3))), (('pick up', 'taxi 1', 'Tamar'), ('move', 'taxi 2', (1, 3))), (('move', 'taxi 1', (3, 1)), ('refuel', 'taxi 2')), (('pick up', 'taxi 1', 'Daniel'), ('move', 'taxi 2', (2, 3))), (('move', 'taxi 1', (3, 2)), ('pick up', 'taxi 2', 'Freyja')), (('drop off', 'taxi 1', 'Tamar'), ('move', 'taxi 2', (2, 4))), (('move', 'taxi 1', (3, 1)), ('drop off', 'taxi 2', 'Freyja')), (('move', 'taxi 1', (2, 1)), ('move', 'taxi 2', (1, 4))), (('drop off', 'taxi 1', 'Daniel'),
('drop off', 'taxi 2', 'Iris'))]

p = TaxiProblem(dicti)
a = p.actions(p.initial)
r = p.initial
for i in range(len(actions)):
    print(actions[i] in a)
    if not actions[i] in a:
        print(actions[i])
        for i in p.map:
            print(i)
        print(r)
        exit()

    r = p.result(r, actions[i])
    a = p.actions(r)
    print(actions[i])
    print(r)'''


#r2 = p.result(p.initial, actions[1])
#print(p.actions(r2))
'''
r3 = p.result(p.initial, (('wait', 'taxi 1'), ('move', 'taxi 2', (0, 0))))
print(p.actions(r1))

r4 = p.result(p.initial, (('wait', 'taxi 1'), ('move', 'taxi 2', (0, 0))))
print(p.actions(r1))

r5 = p.result(p.initial, (('wait', 'taxi 1'), ('move', 'taxi 2', (0, 0))))
print(p.actions(r1))

r6 = p.result(p.initial, (('wait', 'taxi 1'), ('move', 'taxi 2', (0, 0))))
print(p.actions(r1))

# print(dicti)
# print(type(json.dumps(dicti)))
# d = json.dumps(dicti)
# print(json.loads(d))
'''