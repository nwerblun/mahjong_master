from game import *
from queue import PriorityQueue
from score_calculator import Calculator
from math import inf
from functools import total_ordering
from multiprocessing import Process, Pipe
from time import sleep


@total_ordering
class Node:
    def __init__(self, calc, new_deck):
        self.calc = calc
        self.hand = self.calc.hand
        self.remaining_deck = new_deck
        self.prev_dist = inf
        self.hand_value = inf

    def _get_chow_calcs(self, new_calc, tile_name):
        rm_tile = Tile(tile_name)
        possible_chows = new_calc.hand.possible_chows_with_tile(rm_tile)
        calcs = []
        if len(possible_chows) >= 1:
            for p in possible_chows:
                chow_calc = new_calc.make_copy()
                to_rmv = [el for el in p if el != rm_tile]
                for j in to_rmv:
                    chow_calc.hand.concealed_tiles.pop(chow_calc.hand.concealed_tiles.index(j))
                    chow_calc.hand.add_tile_to_hand(True, j.name)
                chow_calc.hand.set_final_tile(tile_name, False)
                calcs += [chow_calc]
            return calcs
        else:
            return None

    def _get_pung_calcs(self, new_calc, tile_name):
        new_calc = new_calc.make_copy()
        rm_tile = Tile(tile_name)
        if new_calc.hand.can_make_pung_with_tile(rm_tile):
            # There are at least 2 of remaining in hand if this is true. Remove them
            new_calc.hand.concealed_tiles.pop(new_calc.hand.concealed_tiles.index(rm_tile))
            new_calc.hand.concealed_tiles.pop(new_calc.hand.concealed_tiles.index(rm_tile))
            # Put the 2 removed tiles + new tile in revealed sets
            new_calc.hand.add_tile_to_hand(True, tile_name)
            new_calc.hand.add_tile_to_hand(True, tile_name)
            new_calc.hand.set_final_tile(tile_name, False)
            return new_calc
        else:
            return None

    def _get_kong_calcs(self, new_calc, tile_name):
        new_calc = new_calc.make_copy()
        rm_tile = Tile(tile_name)
        if new_calc.hand.can_make_kong_with_tile(rm_tile):
            # There are at least 3 of remaining in hand if this is true. Remove them
            new_calc.hand.concealed_tiles.pop(new_calc.hand.concealed_tiles.index(rm_tile))
            new_calc.hand.concealed_tiles.pop(new_calc.hand.concealed_tiles.index(rm_tile))
            new_calc.hand.concealed_tiles.pop(new_calc.hand.concealed_tiles.index(rm_tile))
            # Put the 3 removed tiles + new tile in revealed sets
            new_calc.hand.add_tile_to_hand(True, tile_name)
            new_calc.hand.add_tile_to_hand(True, tile_name)
            new_calc.hand.add_tile_to_hand(True, tile_name)
            new_calc.hand.set_final_tile(tile_name, False)
            return new_calc
        else:
            return None

    def _get_upgraded_kong_calcs(self, new_calc, tile_name):
        new_calc = new_calc.make_copy()
        rm_tile = Tile(tile_name)
        can_upgrade, ind = new_calc.hand.can_upgrade_pung_to_kong_with_tile(rm_tile)
        if can_upgrade:
            new_calc.hand.revealed_tiles.insert(ind, rm_tile)
            return new_calc
        else:
            return None

    def _get_simple_draw_calcs(self, new_calc, tile_name):
        # Draw a new tile 'tile_name' after removing 1 tile.
        new_calc = new_calc.make_copy()
        new_calc.hand.set_final_tile(tile_name, True)
        return new_calc

    def get_neighbors(self, reduced=False):
        neighbors = []
        for i in range(len(self.hand.concealed_tiles)):
            new_calc = self.calc.make_copy()
            discarded = new_calc.hand.concealed_tiles.pop(i)
            for remaining_name, amt in self.remaining_deck:
                if remaining_name == discarded.name:
                    # Don't consider hands where we just throw away the tile we drew
                    continue
                new_deck = self.remaining_deck[:]
                ind = Tile.valid_tile_names.index(remaining_name)
                new_deck[ind] = (new_deck[ind][0], max(0, new_deck[ind][1] - 1))
                if amt > 0:
                    simple_calcs = self._get_simple_draw_calcs(new_calc, remaining_name)
                    pung_calcs = self._get_pung_calcs(new_calc, remaining_name)
                    kong_calcs = self._get_kong_calcs(new_calc, remaining_name)
                    upgraded_calcs = self._get_upgraded_kong_calcs(new_calc, remaining_name)
                    chow_calcs = self._get_chow_calcs(new_calc, remaining_name)

                    if reduced:
                        cond = simple_calcs and any([pung_calcs, kong_calcs, upgraded_calcs, chow_calcs])
                    else:
                        cond = simple_calcs

                    if cond:
                        neighbors += [Node(simple_calcs, new_deck)]
                    if pung_calcs:
                        neighbors += [Node(pung_calcs, new_deck)]
                    if kong_calcs:
                        neighbors += [Node(kong_calcs, new_deck)]
                    if upgraded_calcs:
                        # You don't lose a tile when you upgrade to a kong
                        upgraded_calcs.hand.add_tile_to_hand(False, discarded.name)
                        neighbors += [Node(upgraded_calcs, new_deck)]
                    if chow_calcs:
                        neighbors += [Node(c, new_deck) for c in chow_calcs]
        return neighbors

    def _get_score(self):
        if self.hand_value == inf:
            value, _ = self.calc.get_score_summary()
            self.hand_value = value
            self.calc.pwh = None  # Save memory. Thanks garbage collector!
        return self.hand_value

    def is_goal(self):
        return self._get_score() >= 8

    def heuristic(self):
        # Estimated distance is some factor of current point value to num tiles needed to win
        # TODO: Maybe make a smarter heuristic
        self.calc.pwh = PossibleWinningHand(self.calc.hand)
        if len(self.calc.pwh.four_set_pair_hands) == 0:
            h = 8
        else:
            h = 8 - self._get_score()
        return h

    def __eq__(self, other):
        return self.hand == other.hand

    def __lt__(self, other):
        return self._get_score() < other._get_score()


class Pathfinder:
    def __init__(self, calc):
        self.starting_calc = calc
        remaining_deck = list(zip(Tile.valid_tile_names[:], [4]*len(Tile.valid_tile_names)))
        # tiles in hand cannot be drawn again
        for t in self.starting_calc.hand.concealed_tiles:
            ind = Tile.valid_tile_names.index(t.name)
            remaining_deck[ind] = (remaining_deck[ind][0], max(0, remaining_deck[ind][1]-1))
        self.start_node = Node(calc, remaining_deck)

    def ready_to_check(self):
        return self.starting_calc.hand.get_num_tiles_in_hand() >= 14

    @staticmethod
    def _worker_task(curr, nodes_to_ignore, queue, neighbor):
        if any([curr == i for i in nodes_to_ignore]):
            return
        heuristic_result = neighbor.heuristic()
        prior_before_update = neighbor.prev_dist + heuristic_result
        prior_after_update = curr.prev_dist + 1 + heuristic_result
        neighbor.priority = prior_before_update
        if prior_after_update < prior_before_update and neighbor != curr:
            neighbor.prev_dist = curr.prev_dist + 1
            neighbor.priority = prior_after_update
            queue.put((prior_after_update, neighbor))

    def _a_star(self, pipe_conn, num_wins_req):
        q = PriorityQueue()

        self.start_node.prev_dist = 0
        nodes_to_ignore = []
        iters = 0
        q.put((0, self.start_node))

        curr = None
        num_wins = 0
        winners = []
        while not q.empty():
            # get returns a tuple of (priority, data)
            curr = q.get()[1]
            if curr.is_goal():
                print("Found a solution on iteration", str(iters))
                if q.empty():
                    print("First node was the goal. Quitting.")
                    # If we already won there's no point
                    pipe_conn.send([curr])
                    return
                if not any([curr == i for i in winners]):
                    num_wins += 1
                    winners += [curr.calc]
                    nodes_to_ignore += [curr]
                if num_wins >= num_wins_req:
                    print("Found", str(num_wins_req), "wins. Exiting.")
                    break
            if iters >= 200:
                print("Max iterations in hand-solving exceeded.")
                pipe_conn.send([])

            # Disable for now
            if iters < 3:
                neighbors = curr.get_neighbors(reduced=True)
            else:
                neighbors = curr.get_neighbors(reduced=False)

            for n in neighbors:
                self._worker_task(curr, nodes_to_ignore, q, n)

            nodes_to_ignore += [curr]
            iters += 1

        pipe_conn.send(winners)

    def get_n_fastest_wins(self, n=1):
        rcv, send = Pipe(False)
        p = Process(target=self._a_star, args=(send, n))
        p.start()
        return rcv, p

