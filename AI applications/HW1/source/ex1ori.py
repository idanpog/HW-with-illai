import search
import random
import math
import itertools
import copy
import json


ids = ["211803234", "212641229"]


def zero_max(lst):
    return max(lst) if lst else 0


def zero_min(lst):
    return min(lst) if lst else 0


def not_crashing(action, taxis):
    end_location = [tup[2] for tup in action if tup[0] == "moved"]
    end_location.extend([tuple(taxis[tup[1]]['location'])
                         for tup in action if tup[0] != "moved"])
    return len(end_location) == len(set(end_location))


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""

        """
        @param wait_pickup_passengers passengers waiting for taxi.
        @param taxis dict(taxi_name : dict(location, fuel, capaticy, current_passengers : dict(name : destination))).

        """
        self.map = initial["map"]
        self.n_rows = len(self.map)
        self.n_cols = len(self.map[0])

        g = {}
        for cur_y in range(self.n_rows):
            for cur_x in range(self.n_cols):
                moving_x = [(cur_y, x) for x in [
                    cur_x-1, cur_x+1] if (x >= 0 and x < self.n_cols and self.map[cur_y][x] != 'I')]
                moving_y = [(y, cur_x) for y in [
                    cur_y-1, cur_y+1] if (y >= 0 and y < self.n_rows and self.map[y][cur_x] != 'I')]
                g[(cur_y, cur_x)] = moving_x+moving_y

        taxis = copy.deepcopy(initial["taxis"])
        wait_pickup_passengers = copy.deepcopy(initial["passengers"])
        for taxi in taxis.values():
            taxi["current_passengers"] = []
            taxi["max_fuel"] = taxi["fuel"]

        initial_state = (json.dumps(taxis), json.dumps(wait_pickup_passengers),
                         len(wait_pickup_passengers))
        for pas_info in wait_pickup_passengers.values():
            y_init, x_init = pas_info["location"]
            y_end, x_end = pas_info["destination"]
            if self.map[y_init][x_init] == 'I' or self.map[y_end][x_end] == 'I':
                initial_state = (json.dumps({}), json.dumps({}),
                                 len(wait_pickup_passengers))

        for pass_info in wait_pickup_passengers.values():
            start_loc = pass_info["location"]
            end_locs = [taxi_info["location"] for taxi_info in taxis.values()]
            if not bfs(g, start_loc, end_locs) or not bfs(g, start_loc, [pass_info["destination"]]):
                initial_state = (json.dumps({}), json.dumps({}),
                                 len(wait_pickup_passengers))
                break

        self.n_passengers = len(wait_pickup_passengers)
        self.n_taxis = len(taxis)
        # the third is number of passengers that have not yet been dropped of

        search.Problem.__init__(self, initial_state)

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        map = self.map
        taxis = json.loads(state[0])
        wait_pickup_passengers = json.loads(state[1])
        taxi_actions = {}
        for taxi_name, taxi_info in taxis.items():
            cur_y, cur_x = taxi_info["location"]
            fuel = taxi_info["fuel"]
            capacity = taxi_info["capacity"]
            cur_passengers = taxi_info["current_passengers"]

            ##### wait #####
            taxi_actions[taxi_name] = [("wait", taxi_name)]

            ##### move #####
            if fuel != 0:  # check where the taxi can move
                moving_x = [("move", taxi_name, (cur_y, x)) for x in [
                    cur_x-1, cur_x+1] if (x >= 0 and x < self.n_cols and map[cur_y][x] != 'I')]
                moving_y = [("move", taxi_name, (y, cur_x)) for y in [
                    cur_y-1, cur_y+1] if (y >= 0 and y < self.n_rows and map[y][cur_x] != 'I')]

                taxi_actions[taxi_name].extend(moving_x)
                taxi_actions[taxi_name].extend(moving_y)

            ##### pick up #####
            picking_up = []

            if len(cur_passengers) < capacity:
                for passenger_name, passenger_info in wait_pickup_passengers.items():
                    ploc = passenger_info["location"]
                    if [cur_y, cur_x] == ploc:
                        picking_up.append(
                            ("pick up", taxi_name, passenger_name))
            taxi_actions[taxi_name].extend(picking_up)

            #### drop off ####
            drop_off = [("drop off", taxi_name, pas_name)
                        for (pas_name, pas_dest) in cur_passengers if pas_dest == [cur_y, cur_x]]
            taxi_actions[taxi_name].extend(drop_off)

            ### refuel ###
            if self.map[cur_y][cur_x] == 'G':
                taxi_actions[taxi_name].append(("refuel", taxi_name))

        return filter(lambda x: not_crashing(x, taxis), itertools.product(*taxi_actions.values()))

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        taxis = json.loads(state[0])
        wait_pickup_passengers = json.loads(state[1])
        not_dropped = state[2]
        for tup in action:
            command = tup[0]
            taxi_name = tup[1]
            taxi_dict = taxis[taxi_name]
            if command == "move":
                location = tup[2]
                taxi_dict["location"] = location
                taxi_dict["fuel"] -= 1
            elif command == "pick up":
                pas_name = tup[2]
                taxi_dict["current_passengers"].append(
                    (tup[2], wait_pickup_passengers[pas_name]["destination"]))
                del wait_pickup_passengers[pas_name]
            elif command == "refuel":
                taxi_dict["fuel"] = taxi_dict["max_fuel"]
            elif command == "drop off":
                pas_name = tup[2]
                taxi_dict["current_passengers"] = list(
                    filter(lambda x: x[0] != pas_name, taxi_dict["current_passengers"]))
                not_dropped -= 1
            elif command == "wait":
                pass
            else:
                raise ValueError(f"Not a proper command: {command}")

        return (json.dumps(taxis), json.dumps(wait_pickup_passengers), not_dropped)

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        return not state[2]

    def h(self, node):
        if self.n_taxis == 1:
            return self.h_max(node)
        return max(self.h_ori(node), self.h_max(node))

    def h_max(self, node):
        state = node.state
        wait_pickup_passengers = json.loads(state[1])
        taxis = json.loads(state[0])
        dist_pass_out = {pname: manhattan_dist(*p.values())
                         for (pname, p) in wait_pickup_passengers.items()}
        dist_pass_to = {pas_name: zero_min([manhattan_dist(pas_info['location'], taxi_info['location']) for taxi_info in taxis.values()])
                        for (pas_name, pas_info) in wait_pickup_passengers.items()}
        wait_passengers = [dist_pass_to[p_name] + dist_pass_out[p_name]
                           for p_name in wait_pickup_passengers.keys()]
        dist_pass_in = [zero_max([manhattan_dist(taxi_info["location"], dest) for (
            _, dest) in taxi_info["current_passengers"]]) for taxi_info in taxis.values()]

        return max(zero_max(wait_passengers), zero_max(dist_pass_in))

    def h_ori(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        state = node.state
        wait_pickup_passengers = json.loads(state[1])
        taxis = json.loads(state[0])
        if not state[2]:
            return 0

        ## getting the passengers to the destination ##
        # not picked up
        dist_pass_out = [manhattan_dist(*p.values())
                         for p in wait_pickup_passengers.values()]
        sum_dist_p2l = zero_max(dist_pass_out)
        max_dist_pass = zero_max(dist_pass_out)

        # picked up
        dist_pass_in = [zero_max([manhattan_dist(taxi_info["location"], dest) for (
            _, dest) in taxi_info["current_passengers"]]) for taxi_info in taxis.values()]

        max_dist_taxi = zero_max(dist_pass_in)

        ## picking up and dropping of ##
        number_to_pick = len(dist_pass_out)
        number_to_drop = state[2]
        ## picking up passengers ##
        dist_t2p = [zero_min([manhattan_dist(pas_info['location'], taxi_info['location']) for taxi_info in taxis.values()])
                    for pas_info in wait_pickup_passengers.values()]
        # if self.n_taxis >= 2 and self.n_passengers > 2:
        #    return max(max_dist_pass, max_dist_taxi) + zero_max(dist_t2p) + 2*len(dist_pass_out)/self.n_taxis

        return (max(sum_dist_p2l, sum(dist_t2p)) + number_to_drop+number_to_pick)/self.n_taxis

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        state = node.state
        taxis = json.loads(state[0])
        wait_pickup_passengers = json.loads(state[1])
        num_taxis = self.n_taxis
        num_unpickedup_pass = len(wait_pickup_passengers)
        num_picked_yet_undelivered = state[2] - num_unpickedup_pass

        return (num_unpickedup_pass * 2 + num_picked_yet_undelivered) / num_taxis

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic

        wait_pickup_passengers = dict(name : dict(location : tup, destination : tup))
        """
        state = node.state
        taxis = json.loads(state[0])
        wait_pickup_passengers = json.loads(state[1])
        sum_D = sum([manhattan_dist(*p.values())
                     for p in wait_pickup_passengers.values()])

        sum_T = sum(manhattan_dist(taxi_info["location"], dest) for taxi_info in taxis.values(
        ) for (_, dest) in taxi_info["current_passengers"])

        return (sum_D + sum_T) / self.n_taxis

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


def create_taxi_problem(game):
    return TaxiProblem(game)


def manhattan_dist(u, v):
    return sum([abs(x - y) for x, y in zip(u, v)])
