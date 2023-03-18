from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hands import MahjongHands
from PIL import Image, ImageTk
from functools import total_ordering
from utilities import flatten_list
# contains stuff that is useful in many files about the game.


bamboo_id_ctr = [0] * 9
char_id_ctr = [0] * 9
dots_id_ctr = [0] * 9
dr_id_ctr = {"g": 0, "r": 0, "w": 0}
wind_id_ctr = {"e": 0, "n": 0, "s": 0, "w": 0}


@total_ordering
class Tile:
    def __init__(self, name):
        global bamboo_id_ctr, char_id_ctr, dots_id_ctr, dr_id_ctr, wind_id_ctr
        self.name = name
        self.picture_link = self.get_picture_link_from_name()
        self.img = Image.open(self.picture_link)
        self.ph = ImageTk.PhotoImage(self.img)
        if self.name[0] == "b":
            self.type = "bamboo"
            self.id = bamboo_id_ctr[int(self.name[1])-1]
            bamboo_id_ctr[int(self.name[1])-1] += 1
        elif self.name[0] == "c":
            self.type = "character"
            self.id = char_id_ctr[int(self.name[1])-1]
            char_id_ctr[int(self.name[1])-1] += 1
        elif self.name[0] == "d" and (not self.name[1] == "r"):
            self.type = "dot"
            self.id = dots_id_ctr[int(self.name[1])-1]
            dots_id_ctr[int(self.name[1])-1] += 1
        elif self.name[:2] == "dr":
            self.type = "dragon"
            self.id = dr_id_ctr[self.name[2]]
            dr_id_ctr[self.name[2]] += 1
        else:
            self.type = "wind"
            self.id = wind_id_ctr[self.name[1]]
            wind_id_ctr[self.name[1]] += 1

    @staticmethod
    def is_valid_name(name):
        conds = [
            len(name) < 2,
            len(name) > 3
        ]
        return not any(conds)

    def get_picture_link_from_name(self):
        ind = MahjongHands.tile_names.index(self.name)
        return MahjongHands.tile_pic_files[ind]

    def is_wind(self):
        return self.name[0] == "w"

    def is_dragon(self):
        return self.name[:2] == "dr"

    def is_bamboo(self):
        return self.name[0] == "b"

    def is_char(self):
        return self.name[0] == "c"

    def is_dot(self):
        return self.name[0] == "d" and (not self.name[1] == "r")

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

    def is_sequential_to(self, other):
        if self.type == other.type:
            if self.is_dragon() or self.is_wind():
                return False
            else:
                return int(self.get_tile_number()) == int(other.get_tile_number()) + 1
        return False

    def get_next_sequential_tile_name(self):
        if self.is_dragon() or self.is_wind():
            return None
        elif self.get_tile_number() == 9:
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
        return (self.name < other.name) or ((self.name == other.name) and self.id < other.id)

    def __eq__(self, other):
        return (self.name == other.name) and (self.id == other.id)

    def __hash__(self):
        return (self.name + str(self.id)).__hash__()

    def equals(self, other):
        return self.name == other.name


class Hand:
    def __init__(self):
        # Total set of tiles to be given from some other source
        self.concealed_tiles = []
        self.revealed_tiles = []
        self.declared_concealed_kongs = []
        # Concealed tile "sets." It is potential because 111-222-333 would show up in both chow/pung lists.
        self.potential_concealed_pungs = []
        self.potential_concealed_kongs = []
        self.potential_concealed_chows = []
        self.num_suits_used = 0
        self.uses_bamboo = False
        self.uses_chars = False
        self.uses_dots = False
        self.num_winds = 0
        self.num_dragons = 0
        self.drawn_tile = None

    def _update_suits_and_honor_count(self):
        # Assume lists are sorted
        self.num_winds = 0
        self.num_dragons = 0
        self.num_suits_used = 0
        self.uses_bamboo = self.uses_dots = self.uses_chars = False
        all_tiles = self.concealed_tiles + self.revealed_tiles + flatten_list(self.declared_concealed_kongs)
        if self.drawn_tile:
            all_tiles += [self.drawn_tile]

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

    def _hand_contains_concealed_chow_from_starting_tile(self, start_tile):
        next_tile_name = Tile.tile_name_to_next_tile_name(start_tile.name)
        if next_tile_name is None:
            return False
        next_next_tile_name = Tile.tile_name_to_next_tile_name(next_tile_name)
        tile_names = [t.name for t in self.concealed_tiles]
        return (next_tile_name in tile_names) and (next_next_tile_name in tile_names)

    def get_num_honor_tiles(self):
        return self.num_dragons + self.num_winds

    def set_drawn_tile(self, name):
        if Tile.is_valid_name(name):
            self.drawn_tile = Tile(name)
        self._update_hand()

    def get_num_tiles_in_hand(self):
        return len(flatten_list(self.declared_concealed_kongs)) + len(self.concealed_tiles) + len(self.revealed_tiles)

    def clear_hand(self):
        self.__init__()

    def sort_hand(self):
        if len(self.concealed_tiles) > 1:
            self.concealed_tiles = sorted(self.concealed_tiles)
        if len(self.declared_concealed_kongs) >= 1:
            self.declared_concealed_kongs = sorted(self.declared_concealed_kongs)

    def _update_hand(self):
        self.sort_hand()
        self._update_suits_and_honor_count()

    def add_tile_to_hand(self, revealed, tile_name, more=False):
        # More is an optimization parameter. Set to true if you expect to continue adding pieces.
        # Setting the drawn tile will force an update, or call add tiles again with tile_name = None
        if self.get_num_tiles_in_hand() >= 16 or tile_name is None:
            self._update_hand()
            return
        if revealed:
            self.revealed_tiles += [Tile(tile_name)]
        else:
            self.concealed_tiles += [Tile(tile_name)]
        if not more:
            self._update_hand()

    def add_declared_concealed_kong_to_hand(self, tile_name):
        self.declared_concealed_kongs += [[Tile(tile_name), Tile(tile_name), Tile(tile_name), Tile(tile_name)]]
        self._update_suits_and_honor_count()

    def add_revealed_kong_to_hand(self, tile_name):
        # Not [Tile(tile_name)] * 4 because I don't want the same reference 4 times, but 4 different objects
        self.revealed_tiles += [Tile(tile_name), Tile(tile_name), Tile(tile_name), Tile(tile_name)]

    def is_fully_concealed(self):
        return len(self.revealed_tiles) == 0

    def get_revealed_sets(self):
        # User is probably still typing
        if len(self.revealed_tiles) < 3:
            return [], [], []
        temp_set = self.revealed_tiles[:]
        revealed_chows = []
        revealed_pungs = []
        revealed_kongs = []
        if len(temp_set) == 3:
            t0, t1, t2 = temp_set[0], temp_set[1], temp_set[2]
            if t2.is_sequential_to(t1) and t1.is_sequential_to(t0):
                revealed_chows += [[t0, t1, t2]]
            else:
                revealed_pungs += [[t0, t1, t2]]
        elif len(temp_set) > 3:
            while len(temp_set) > 0:
                t0, t1, t2, t3 = temp_set[0], temp_set[1], temp_set[2], temp_set[3]
                if t0.equals(t1) and t1.equals(t2) and t2.equals(t3):
                    revealed_kongs += [[t0, t1, t2, t3]]
                    temp_set = temp_set[4:]
                elif t0.equals(t1) and t1.equals(t2) and not t2.equals(t3):
                    revealed_pungs += [[t0, t1, t2]]
                    temp_set = temp_set[3:]
        return revealed_chows, revealed_pungs, revealed_kongs

