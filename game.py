from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hands import MahjongHands
from PIL import Image, ImageTk
from functools import total_ordering
# contains stuff that is useful in many files about the game.


@total_ordering
class Tile:
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

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
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

        self.sets = []

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
        all_tiles = self.concealed_tiles + self.revealed_tiles
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

        for tile in self.declared_concealed_kongs:
            if tile.is_wind():
                self.num_winds += 4
            elif tile.is_dragon():
                self.num_dragons += 4
            elif not self.uses_bamboo and tile.is_bamboo():
                self.num_suits_used += 1
                self.uses_bamboo = True
            elif not self.uses_chars and tile.is_char():
                self.num_suits_used += 1
                self.uses_chars = True
            elif not self.uses_dots and tile.is_dot():
                self.num_suits_used += 1
                self.uses_dots = True

    def _count_potential_concealed_pungs_kongs(self):
        # Revealed tiles must be sets, so it's either a pung, kong or chow guaranteed. Don't need to check.
        self.potential_concealed_pungs = []
        self.potential_concealed_kongs = []
        all_tiles = self.concealed_tiles
        if self.drawn_tile:
            all_tiles += [self.drawn_tile]
        all_unique_tiles = list(set(all_tiles))  # Don't want to triple count tiles. Only check unique tiles.
        for tile in all_unique_tiles:
            amt = all_tiles.count(tile)
            if amt == 3:
                self.potential_concealed_pungs += [tile]
            elif amt == 4:
                self.potential_concealed_kongs += [tile]

    def _count_potential_concealed_chows(self):
        self.potential_concealed_chows = []
        all_tiles = self.concealed_tiles
        if self.drawn_tile:
            all_tiles += [self.drawn_tile]
        all_tile_names = [t.name for t in all_tiles]
        for tile in all_tiles:
            next_tile_name = tile.get_next_sequential_tile_name()
            if next_tile_name in all_tile_names:
                next_tile = all_tiles[all_tile_names.index(next_tile_name)]
                next_next_tile_name = next_tile.get_next_sequential_tile()
                if next_next_tile_name in all_tile_names:
                    next_next_tile = all_tiles[all_tile_names.index(next_next_tile_name)]
                    self.potential_concealed_chows += [[tile, next_tile, next_next_tile]]

    def get_num_honor_tiles(self):
        return self.num_dragons + self.num_winds

    def discard_tile(self, tile_name, discard_draw=False):
        # If discard draw is true, tile name is ignored
        if discard_draw:
            self.drawn_tile = None

    def draw_tile(self, name):
        self.drawn_tile = Tile(name)

    def add_tile_to_hand(self, revealed, tile_name):
        if revealed:
            self.revealed_tiles += [Tile(tile_name)]
        else:
            self.concealed_tiles += [Tile(tile_name)]

        if len(self.concealed_tiles) > 1:
            self.concealed_tiles = sorted(self.concealed_tiles)
        if len(self.revealed_tiles) > 1:
            self.revealed_tiles = sorted(self.revealed_tiles)

    def declare_concealed_kong(self):
        pass

    def upgrade_revealed_pung(self):
        pass

    def is_fully_concealed(self):
        return len(self.revealed_tiles) == 0

    def get_num_chows(self):
        pass

    def get_num_pungs(self):
        pass

    def get_num_kongs(self):
        pass

    def get_num_sets(self):
        pass
