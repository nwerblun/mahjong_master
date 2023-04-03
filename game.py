from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hands import MahjongHands
from PIL import Image, ImageTk
from functools import total_ordering
from utilities import flatten_list


@total_ordering
class Tile:
    valid_tile_names = [
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "b8",
        "b9",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
        "c8",
        "c9",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "d8",
        "d9",
        "drg",
        "drr",
        "drw",
        "we",
        "wn",
        "ws",
        "ww"
    ]

    def __init__(self, name):
        self.name = name
        self.picture_link = self.get_picture_link_from_name()
        self.img = Image.open(self.picture_link)
        self.ph = ImageTk.PhotoImage(self.img)
        if self.name[0] == "b":
            self.type = "bamboo"
        elif self.name[0] == "c":
            self.type = "character"
        elif self.name[0] == "d" and (not self.name[1] == "r"):
            self.type = "dot"
        elif self.name[:2] == "dr":
            self.type = "dragon"
        else:
            self.type = "wind"

    @staticmethod
    def is_valid_name(name):
        return name in Tile.valid_tile_names

    def get_picture_link_from_name(self):
        ind = MahjongHands.tile_names.index(self.name)
        return MahjongHands.tile_pic_files[ind]

    def is_wind(self):
        return self.type == "wind"

    def is_dragon(self):
        return self.type == "dragon"

    def is_bamboo(self):
        return self.type == "bamboo"

    def is_char(self):
        return self.type == "character"

    def is_dot(self):
        return self.type == "dot"

    def get_dragon_type(self):
        if not self.is_dragon():
            return None
        return self.name[2]

    def get_wind_direction(self):
        if not self.is_wind():
            return None
        return self.name[1]

    def get_tile_number(self):
        if self.is_dragon() or self.is_wind():
            return None
        return self.name[1]

    def is_sequential_to(self, other, ignore_suit=False):
        if self.type == other.type and (not ignore_suit):
            if self.is_dragon() or self.is_wind():
                return False
            else:
                return int(self.get_tile_number()) == int(other.get_tile_number()) + 1
        elif ignore_suit:
            if self.is_dragon() or self.is_wind():
                return False
            else:
                return int(self.get_tile_number()) == int(other.get_tile_number()) + 1
        return False

    def get_next_sequential_tile_name(self):
        if self.is_dragon() or self.is_wind():
            return None
        elif self.get_tile_number() == "9":
            return None
        return self.name[0] + str(int(self.get_tile_number()) + 1)

    @staticmethod
    def tile_name_to_next_tile_name(name):
        if name is None or name[:2] == "dr" or name[0] == "w":
            return None
        elif int(name[1]) == 9:
            return None
        else:
            return name[0] + str(int(name[1]) + 1)

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class VoidTile(Tile):
    def __init__(self):
        super().__init__("z1")
        self.type = "void"

    def get_picture_link_from_name(self):
        return "./img/base_tiles/z1.png"

    def get_next_sequential_tile_name(self):
        return None

    @staticmethod
    def tile_name_to_next_tile_name(name):
        return None

    def is_sequential_to(self, other, ignore_suit=False):
        return False


class Hand:
    def __init__(self):
        # Total set of tiles to be given from some other source
        self.concealed_tiles = []
        self.revealed_tiles = []
        self.declared_concealed_kongs = []
        self.num_suits_used = 0
        self.uses_bamboo = False
        self.uses_chars = False
        self.uses_dots = False
        self.final_tile = None
        self.self_drawn_final_tile = False
        self.num_winds = 0
        self.num_dragons = 0

    def _hand_is_legal(self):
        if self.get_num_tiles_in_hand() < 14 or self.get_num_tiles_in_hand() > 18:
            return False
        total_tile_list = self.concealed_tiles + self.revealed_tiles + self.declared_concealed_kongs
        total_tile_list = flatten_list(total_tile_list)
        for t in total_tile_list:
            if total_tile_list.count(t) > 4:
                return False
        return True

    def _update_suits_and_honor_count(self):
        # Assume lists are sorted
        self.num_winds = 0
        self.num_dragons = 0
        self.num_suits_used = 0
        self.uses_bamboo = self.uses_dots = self.uses_chars = False
        all_tiles = self.concealed_tiles + self.revealed_tiles + flatten_list(self.declared_concealed_kongs)

        for tile in all_tiles:
            if tile.is_wind():
                self.num_winds += 1
            elif tile.is_dragon():
                self.num_dragons += 1
            elif not self.uses_bamboo and tile.is_bamboo():
                self.num_suits_used += 1
                self.uses_bamboo = True
            elif not self.uses_chars and tile.is_char():
                self.num_suits_used += 1
                self.uses_chars = True
            elif not self.uses_dots and tile.is_dot():
                self.num_suits_used += 1
                self.uses_dots = True

    def get_num_honor_tiles(self):
        return self.num_dragons + self.num_winds

    def get_num_dragons(self):
        return self.num_dragons

    def get_num_winds(self):
        return self.num_winds

    def set_final_tile(self, name, self_drawn):
        if Tile.is_valid_name(name):
            self.final_tile = Tile(name)
            self.self_drawn_final_tile = self_drawn
            if self_drawn:
                self.concealed_tiles += [self.final_tile]
            else:
                self.revealed_tiles += [self.final_tile]
        self._update_hand()

    def get_num_tiles_in_hand(self):
        return len(flatten_list(self.declared_concealed_kongs))\
            + len(self.concealed_tiles) + len(self.revealed_tiles)

    def clear_hand(self):
        self.__init__()

    def _sort_hand(self):
        if len(self.concealed_tiles) > 1:
            self.concealed_tiles = sorted(self.concealed_tiles)
        if len(self.declared_concealed_kongs) >= 1:
            self.declared_concealed_kongs = sorted(self.declared_concealed_kongs)
        if len(self.revealed_tiles) > 1:
            self.revealed_tiles  = sorted(self.revealed_tiles)

    def _update_hand(self):
        self._sort_hand()
        self._update_suits_and_honor_count()

    def add_tile_to_hand(self, revealed, tile_name, more=False):
        # More is an optimization parameter. Set to true if you expect to continue adding pieces.
        # Setting the drawn tile will force an update, or call add tiles again with tile_name = None
        if self.get_num_tiles_in_hand() >= 16 or tile_name is None:
            self._update_hand()
            return
        if revealed and (tile_name in Tile.valid_tile_names):
            self.revealed_tiles += [Tile(tile_name)]
        elif tile_name in Tile.valid_tile_names:
            self.concealed_tiles += [Tile(tile_name)]
        if not more:
            self._update_hand()

    def add_declared_concealed_kong_to_hand(self, tile_name):
        if tile_name in Tile.valid_tile_names:
            self.declared_concealed_kongs += [[Tile(tile_name), Tile(tile_name), Tile(tile_name), Tile(tile_name)]]
            self._update_suits_and_honor_count()

    def add_revealed_kong_to_hand(self, tile_name):
        if tile_name in Tile.valid_tile_names:
            # Not [Tile(tile_name)] * 4 because I don't want the same reference 4 times, but 4 different objects
            self.revealed_tiles += [Tile(tile_name), Tile(tile_name), Tile(tile_name), Tile(tile_name)]

    def is_fully_concealed(self):
        # Fully concealed up to the final tile.
        if not self.self_drawn_final_tile:
            return len(self.revealed_tiles) <= 1
        return not len(self.revealed_tiles)

    def get_revealed_sets(self):
        # Assume
        # User is probably still typing
        if len(self.revealed_tiles) < 3:
            return [], [], []
        temp_set = self.revealed_tiles[:]
        revealed_chows = []
        revealed_pungs = []
        revealed_kongs = []
        while len(temp_set) >= 3:
            t0, t1, t2 = temp_set[0], temp_set[1], temp_set[2]
            if len(temp_set) > 3:
                t3 = temp_set[3]
            else:
                t3 = VoidTile()
            if t0 == t1 and t1 == t2 and t2 != t3:
                revealed_pungs += [[t0, t1, t2]]
                temp_set = temp_set[3:]
            elif t2.is_sequential_to(t1) and t1.is_sequential_to(t0):
                revealed_chows += [[t0, t1, t2]]
                temp_set = temp_set[3:]
            elif t0 == t1 and t1 == t2 and t2 == t3:
                revealed_kongs += [[t0, t1, t2, t3]]
                temp_set = temp_set[4:]
            else:
                return [], [], []

        return revealed_chows, revealed_pungs, revealed_kongs

    def get_num_revealed_sets(self, kind="all"):
        revealed_sets = self.get_revealed_sets()
        if kind == "all":
            return len(revealed_sets[0]) + len(revealed_sets[1]) + len(revealed_sets[2])
        elif kind == "chow":
            return len(revealed_sets[0])
        elif kind == "pung":
            return len(revealed_sets[1])
        elif kind == "kong":
            return len(revealed_sets[2])
        else:
            return -1


class PossibleWinningHand(Hand):
    def __init__(self, hand):
        super().__init__()
        self.revealed_tiles = hand.revealed_tiles[:]
        self.concealed_tiles = hand.concealed_tiles[:]
        self.declared_concealed_kongs = hand.declared_concealed_kongs[:]
        self.point_conditions = [0] * len(MahjongHands.get_hand_titles())
        self.four_set_pair_hands = []
        self.closed_wait = False
        self.single_wait = False
        self.edge_wait = False
        if hand.final_tile:
            self.set_final_tile(hand.final_tile.name, hand.self_drawn_final_tile)
        self._update_hand()
        self._construct_four_set_pair_hands()

    def _get_base_dict(self):
        rc, rp, rk = self.get_revealed_sets()
        d = {
            "revealed_chows": rc,
            "revealed_pungs": rp,
            "revealed_kongs": rk,
            "concealed_chows": [],
            "concealed_pungs": [],
            "concealed_kongs": [],
            "declared_concealed_kongs": self.declared_concealed_kongs,
            "pair": [],
            "discard": [],
            "knitted_straight": False,
            "point_conditions": self.point_conditions[:]
        }
        return d

    @staticmethod
    def _has_knitted_straight(tile_list):
        all_tiles = tile_list[:]
        knitted_set_variants = [
            [
                Tile("b1"), Tile("b4"), Tile("b7"),
                Tile("c2"), Tile("c5"), Tile("c8"),
                Tile("d3"), Tile("d6"), Tile("d9")
            ],
            [
                Tile("b1"), Tile("b4"), Tile("b7"),
                Tile("d2"), Tile("d5"), Tile("d8"),
                Tile("c3"), Tile("c6"), Tile("c9")
            ],
            [
                Tile("c1"), Tile("c4"), Tile("c7"),
                Tile("d2"), Tile("d5"), Tile("d8"),
                Tile("b3"), Tile("b6"), Tile("b9")
            ],
            [
                Tile("c1"), Tile("c4"), Tile("c7"),
                Tile("b2"), Tile("b5"), Tile("b8"),
                Tile("d3"), Tile("d6"), Tile("d9")
            ],
            [
                Tile("d1"), Tile("d4"), Tile("d7"),
                Tile("c2"), Tile("c5"), Tile("c8"),
                Tile("b3"), Tile("b6"), Tile("b9")
            ],
            [
                Tile("d1"), Tile("d4"), Tile("d7"),
                Tile("b2"), Tile("b5"), Tile("b8"),
                Tile("c3"), Tile("c6"), Tile("c9")
            ]
        ]
        straight_exists = False
        leftover_concealed = []
        for variant in knitted_set_variants:
            uniques = set(all_tiles)
            if len(uniques.intersection(set(variant))) == len(variant):
                straight_exists = True
                leftover_concealed = [t for t in all_tiles if t not in variant]
        return straight_exists, leftover_concealed

    def _is_single_wait(self, t):
        all_tiles = self.concealed_tiles + self.revealed_tiles + flatten_list(self.declared_concealed_kongs)
        if all_tiles.count(t) == 0:
            return False
        all_tiles.pop(all_tiles.index(t))
        groups = self._group_into_sets(all_tiles)
        for group in groups:
            temp_all_tiles = all_tiles[:]
            for t in flatten_list(group):
                temp_all_tiles.pop(temp_all_tiles.index(t))
            if len(temp_all_tiles) == 0:
                return True
        return False

    # TODO: Fix. Group into sets is too unpredictable.... Or is it?
    def _is_edge_wait(self, t):
        if t.get_tile_number() not in ["3", "7"]:
            return False
        all_tiles = self.concealed_tiles[:]
        if t.get_tile_number() == "3":
            one = Tile(t.name[0] + "1")
            two = Tile(t.name[0] + "2")
        else:
            one = Tile(t.name[0] + "8")
            two = Tile(t.name[0] + "9")
        if all_tiles.count(one) >= 1 and all_tiles.count(two) >= 1:
            all_tiles.pop(all_tiles.index(one))
            all_tiles.pop(all_tiles.index(two))
        else:
            return False
        groups = self._group_into_sets(all_tiles)
        # For each grouping in groups
        for group in groups:
            temp_all_tiles = all_tiles[:]
            for t in flatten_list(group):
                temp_all_tiles.pop(temp_all_tiles.index(t))
            if len(temp_all_tiles) == 2 and temp_all_tiles[0] == temp_all_tiles[1]:
                return True
        return False

    # TODO: Fix. Group into sets is too unpredictable.... Or is it?
    def _is_closed_wait(self, t):
        if t.get_tile_number() not in ["2", "3", "4", "5", "6", "7", "8"]:
            return False
        all_tiles = self.concealed_tiles[:]
        prev_t = Tile(t.name[0] + str(int(t.get_tile_number()) - 1))
        next_t = Tile(t.name[0] + str(int(t.get_tile_number()) + 1))
        if all_tiles.count(prev_t) >= 1 and all_tiles.count(next_t) >= 1:
            all_tiles.pop(all_tiles.index(prev_t))
            all_tiles.pop(all_tiles.index(next_t))
        else:
            return False
        groups = self._group_into_sets(all_tiles)
        for group in groups:
            temp_all_tiles = all_tiles[:]
            for t in flatten_list(group):
                temp_all_tiles.pop(temp_all_tiles.index(t))
            if len(temp_all_tiles) == 2 and temp_all_tiles[0] == temp_all_tiles[1]:
                return True
        return False

    def _get_waits(self):
        all_tiles = self.concealed_tiles + self.revealed_tiles + flatten_list(self.declared_concealed_kongs)
        single_waits = []
        edge_waits = []
        closed_waits = []
        # TODO: Change this to some sort of global 'deck' that keeps track of tiles remaining including discards
        for n in Tile.valid_tile_names:
            temp_tile = Tile(n)
            if all_tiles.count(temp_tile) == 4:
                continue
            if self._is_single_wait(temp_tile):
                single_waits += [temp_tile]
            elif self._is_closed_wait(temp_tile):
                closed_waits += [temp_tile]
            elif self._is_edge_wait(temp_tile):
                edge_waits += [temp_tile]
        return single_waits, closed_waits, edge_waits

    def set_final_tile(self, name, self_drawn):
        self.final_tile = Tile(name)
        self.self_drawn_final_tile = self_drawn
        if self_drawn:
            self.concealed_tiles.pop(self.concealed_tiles.index(self.final_tile))
        else:
            self.revealed_tiles.pop(self.revealed_tiles.index(self.final_tile))
        singles, closed, edges = self._get_waits()
        total_waits = len(singles) + len(closed) + len(edges)
        if total_waits == 1:
            if len(singles) == 1 and singles[0] == self.final_tile:
                self.single_wait = True
                self.edge_wait = False
                self.closed_wait = False
                self.concealed_tiles += [self.final_tile]
            elif len(edges) == 1 and edges[0] == self.final_tile:
                self.single_wait = False
                self.edge_wait = True
                self.closed_wait = False
                if self_drawn:
                    self.concealed_tiles += [self.final_tile]
                else:
                    self.revealed_tiles += [self.final_tile]
            elif len(closed) == 1 and closed[0] == self.final_tile:
                self.single_wait = False
                self.edge_wait = False
                self.closed_wait = True
                if self_drawn:
                    self.concealed_tiles += [self.final_tile]
                else:
                    self.revealed_tiles += [self.final_tile]
            else:
                self.single_wait = False
                self.edge_wait = False
                self.closed_wait = True
                if self_drawn:
                    self.concealed_tiles += [self.final_tile]
                else:
                    self.revealed_tiles += [self.final_tile]
        else:
            self.single_wait = False
            self.edge_wait = False
            self.closed_wait = False
            if self_drawn:
                self.concealed_tiles += [self.final_tile]
            else:
                self.revealed_tiles += [self.final_tile]
        self._update_hand()
        self._construct_four_set_pair_hands()

    def _group_into_sets(self, tiles):
        if len(tiles) < 3:
            return [[]]
        t0 = tiles[0]
        t0_next_name = t0.get_next_sequential_tile_name()
        t0_second = VoidTile() if t0_next_name is None else Tile(t0_next_name)
        t0_next_name = t0_second.get_next_sequential_tile_name()
        t0_third = VoidTile() if t0_next_name is None else Tile(t0_next_name)
        sets_and_remainders = []
        if tiles.count(t0_second) >= 1 and tiles.count(t0_third) >= 1:
            ind_next = tiles.index(t0_second)
            ind_third = tiles.index(t0_third)
            remainder = tiles[:]
            remainder.pop(ind_third)
            remainder.pop(ind_next)
            remainder.pop(0)
            sets_and_remainders += [[
                [t0, tiles[ind_next], tiles[ind_third]],
                remainder
            ]]
        if tiles.count(t0) >= 3:
            temp_tiles = tiles[1:]
            ind_next = temp_tiles.index(t0)
            t0_1 = temp_tiles.pop(ind_next)
            ind_third = temp_tiles.index(t0)
            t0_2 = temp_tiles.pop(ind_third)
            sets_and_remainders += [[
                [t0, t0_1, t0_2],
                temp_tiles
            ]]
        if tiles.count(t0) == 4:
            temp_tiles = tiles[1:]
            ind_next = temp_tiles.index(t0)
            t0_1 = temp_tiles.pop(ind_next)
            ind_third = temp_tiles.index(t0)
            t0_2 = temp_tiles.pop(ind_third)
            ind_fourth = temp_tiles.index(t0)
            t0_3 = temp_tiles.pop(ind_fourth)
            sets_and_remainders += [[
                [t0, t0_1, t0_2, t0_3],
                temp_tiles
            ]]

        if not len(sets_and_remainders):
            return [[]]
        to_be_returned = []
        for starting in sets_and_remainders:
            for result in self._group_into_sets(starting[1]):
                to_be_returned += [[starting[0]] + result]
        return to_be_returned

    def _construct_four_set_pair_hands(self):
        self.four_set_pair_hands = []
        if not self._hand_is_legal():
            return
        sets_remaining = 4 - self.get_num_revealed_sets("all") - len(self.declared_concealed_kongs)
        # Can we construct this many sets with the concealed tiles + drawn tile?
        all_tiles = self.concealed_tiles[:]
        all_tiles = sorted(all_tiles)
        # Generate all possible pairs, and the remaining lists if you take out those pairs
        pairs_list = []
        tiles_minus_pairs = []
        for i in range(len(all_tiles)):
            temp_all_tiles = all_tiles[:]
            if all_tiles.count(all_tiles[i]) >= 2:
                t0 = temp_all_tiles.pop(i)
                if temp_all_tiles.index(all_tiles[i]) == i:
                    t1 = temp_all_tiles.pop(i)
                    pairs_list += [[t0, t1]]
                    tiles_minus_pairs += [temp_all_tiles]
        # Can't make a hand without a pair
        if len(pairs_list) == 0:
            return

        for i, leftover in enumerate(tiles_minus_pairs):
            knitted_check, left_after_straight = self._has_knitted_straight(leftover)
            if knitted_check:
                knitted_sets_remaining = sets_remaining - 3
                s = self._group_into_sets(left_after_straight)
            else:
                s = self._group_into_sets(leftover)
                knitted_sets_remaining = sets_remaining
            for combination in s:
                if len(combination) == sets_remaining or (knitted_check and knitted_sets_remaining == 0):
                    if knitted_check:
                        tile_to_discard = list(set(left_after_straight) - set(flatten_list(combination)))
                    else:
                        tile_to_discard = list(set(leftover) - set(flatten_list(combination)))
                    temp_dict = self._get_base_dict()
                    temp_dict["knitted_straight"] = knitted_check
                    temp_dict["discard"] += tile_to_discard
                    for complete_set in combination:
                        if len(complete_set) == 4:
                            temp_dict["concealed_kongs"] += [complete_set]
                        elif complete_set[0] == complete_set[1]:
                            temp_dict["concealed_pungs"] += [complete_set]
                        else:
                            temp_dict["concealed_chows"] += [complete_set]
                    temp_dict["pair"] = pairs_list[i][:]
                    self.four_set_pair_hands += [temp_dict]
        return
