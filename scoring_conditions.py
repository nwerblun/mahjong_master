from game import *


def lesser_honors_knitted_seq(hand):
    all_tiles = hand.concealed_tiles + [hand.drawn_tile] if hand.drawn_tile else hand.concealed_tiles
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
    all_honor_tiles = [Tile("drg"), Tile("drr"), Tile("drw"), Tile("we"), Tile("wn"), Tile("ws"), Tile("ww")]


def knitted_straight(hand):
    pass


def seven_pairs(hand):
    all_tiles = hand.concealed_tiles + [hand.drawn_tile] if hand.drawn_tile else hand.concealed_tiles
    all_tiles = sorted(all_tiles)
    counts = [all_tiles.count(t) for t in all_tiles]
    fails = [
        counts.count(1) > 1,
        counts.count(3) > 1,
        counts.count(3) == 1 and counts.count(1) >= 1,
    ]
    if any(fails):
        return 0
    return 1


def greater_honors_knitted_tiles(hand):
    pass


def seven_shifted_pairs(hand):
    all_tiles = hand.concealed_tiles + [hand.drawn_tile] if hand.drawn_tile else hand.concealed_tiles
    all_tiles = sorted(all_tiles)
    counts = [all_tiles.count(t) for t in all_tiles]
    if counts.count(1) == 1:
        all_tiles.pop(counts.index(1))

    if not counts.count(2) == len(counts):
        return 0

    uniques = sorted(list(set(all_tiles)), key=lambda x: x.get_tile_number())
    for i in range(1, len(uniques)):
        if not uniques[i].is_sequential_to(uniques[i-1], ignore_suit=True):
            return 0
    return 1


def thirteen_orphans(hand):
    all_tiles = hand.concealed_tiles + [hand.drawn_tile] if hand.drawn_tile else hand.concealed_tiles
    hand_size = len(hand.concealed_tiles) + 1 if hand.drawn_tile else len(hand.concealed_tiles)
    tiles_needed = [
        Tile("b1"), Tile("b9"), Tile("c1"),
        Tile("c9"), Tile("d1"), Tile("d9"),
        Tile("drg"), Tile("drr"), Tile("drw"),
        Tile("we"), Tile("wn"), Tile("ws"),
        Tile("ww")
    ]
    duplicate_tile_possibilities = tiles_needed[:]
    ctr = 0
    while ctr <= 13:
        if all_tiles[0] in tiles_needed:
            tiles_needed.pop(tiles_needed.index(all_tiles[0]))
            all_tiles.pop(0)
        ctr += 1

    if len(tiles_needed) > 0:
        return 0
    if hand_size == 14 and all_tiles[0] in duplicate_tile_possibilities:
        return 1
    if hand_size == 15 and ((all_tiles[0] in duplicate_tile_possibilities)
                            or (all_tiles[1] in duplicate_tile_possibilities)):
        return 1
    return 0


class TileSet:
    def __init__(self, tile_list, set_type, concealed=False, declared=None):
        # Set type = chow pung or kong as a str
        self.used = False
        self.excluded = False
        self.concealed = concealed
        self.declared = declared
        self.set_type = set_type
        self.suit = tile_list[0].type
        # Should be implied but... better safe than sorry
        if tile_list[0].is_wind() and (set_type == "pung" or set_type == "kong"):
            self.numbers = tile_list[0].get_wind_direction()
        elif tile_list[0].is_dragon() and (set_type == "pung" or set_type == "kong"):
            self.numbers = tile_list[0].get_dragon_type()
        elif (set_type == "chow") or (set_type == "pung"):
            self.numbers = tile_list[0].get_tile_number() + \
                           tile_list[1].get_tile_number() + tile_list[2].get_tile_number()

    @staticmethod
    def update_used_excluded_stats(setlist):
        if type(setlist) == TileSet:
            to_update = [setlist]
        else:
            to_update = setlist
        if TileSet.any_used(to_update):
            for tset in to_update:
                if not tset.used:
                    tset.excluded = True
                    tset.used = True
        else:
            for tset in to_update:
                tset.used = True

    @staticmethod
    def any_excluded(setlist):
        return any(map(lambda x: x.excluded, setlist))

    @staticmethod
    def any_used(setlist):
        return any(map(lambda x: x.used, setlist))

    def is_same_sequence_different_suit_as(self, other):
        return (self.suit != other.suit) and (self.numbers == other.numbers) \
            and (self.set_type == "chow") and (other.set_type == "chow")

    def __eq__(self, other):
        return (self.suit == other.suit) and (self.numbers == other.numbers) and (self.set_type == other.set_type)

    def __hash__(self):
        return (self.suit + self.numbers + self.set_type).__hash__()

    def __repr__(self):
        return self.suit + "_" + self.numbers


def pure_double_chow(chows):
    amt = 0
    # Need 1 fresh + 1 used or 2 fresh. 2 fresh -> both used. 1 fresh + 1 used -> fresh becomes excluded.
    # Priority is 1 fresh 1 used > 2 fresh.
    # Use recursion to ensure non-identical principle is not violated (cannot use the same set for the same fan twice)
    fresh_chows = [ts for ts in chows if (not ts.excluded and not ts.used)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    if len(fresh_chows) < 1:
        return 0
    if used_but_not_excluded_chows.count(fresh_chows[0]) >= 1:
        index = used_but_not_excluded_chows.index(fresh_chows[0])
        recursion_list = fresh_chows[1:] + used_but_not_excluded_chows[:index] + used_but_not_excluded_chows[index+1:]
        TileSet.update_used_excluded_stats([fresh_chows[0], used_but_not_excluded_chows[index]])
        extra = pure_double_chow(recursion_list)
        amt += 1 + extra
    # Exclusively >1 because we are not excluding fresh[0]
    elif used_but_not_excluded_chows.count(fresh_chows[0]) == 0 and fresh_chows.count(fresh_chows[0]) > 1:
        index = fresh_chows[1:].index(fresh_chows[0])
        recursion_list = fresh_chows[1:index+1] + fresh_chows[index+2:]
        TileSet.update_used_excluded_stats([fresh_chows[0], fresh_chows[index+1]])
        extra, met = pure_double_chow(recursion_list)
        amt += 1 + extra
    else:
        return 0
    return amt


def mixed_double_chow(chows):
    amt = 0
    # Need 1 fresh + 1 used or 2 fresh. 2 fresh -> both used. 1 fresh + 1 used -> fresh becomes excluded.
    # Priority is 1 fresh 1 used > 2 fresh.
    # Use recursion to ensure non-identical principle is not violated (cannot use the same set for the same fan twice)
    fresh_chows = [ts for ts in chows if (not ts.excluded and not ts.used)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    if len(fresh_chows) < 1:
        return 0
    used_count_map = list(map(lambda x: x.is_same_sequence_different_suit_as(fresh_chows[0]),
                              used_but_not_excluded_chows))
    fresh_count_map = list(map(lambda x: x.is_same_sequence_different_suit_as(fresh_chows[0]),
                               fresh_chows))
    if used_count_map.count(True) >= 1:
        index = used_count_map.index(True)
        recursion_list = fresh_chows[1:] + used_but_not_excluded_chows[:index] + used_but_not_excluded_chows[index + 1:]
        TileSet.update_used_excluded_stats([fresh_chows[0], used_but_not_excluded_chows[index]])
        extra = mixed_double_chow(recursion_list)
        amt += 1 + extra
    # >= 1 because the map will always show False for fresh[0], so 1 is acceptable.
    elif used_count_map.count(True) == 0 and fresh_count_map.count(True) >= 1:
        index = fresh_count_map[1:].index(True)
        recursion_list = fresh_chows[1:index+1] + fresh_chows[index+2:]
        TileSet.update_used_excluded_stats([fresh_chows[0], fresh_chows[index+1]])
        extra = mixed_double_chow(recursion_list)
        amt += 1 + extra
    else:
        return 0
    return amt


def short_straight(chows):
    pass


def two_terminal_chows(chows):
    pass


def terminal_non_dragon_honor_pung(pungs, kongs):
    pass


def melded_kong(kongs):
    pass


def voided_suit():
    raise NotImplemented("Do it in the score calc")


def no_honor_tiles():
    raise NotImplemented("Do it in the score calc")


def self_drawn():
    raise NotImplemented("Do it in the score calc? Maybe?")


def flowers():
    return 0


def edge_wait(chows):
    pass


def closed_wait(chows):
    pass


def single_wait(pair):
    pass


def dragon_pung(pungs, kongs):
    pass


def round_wind_pung(pungs, kongs):
    pass


def seat_wind_pung(pungs, kongs):
    pass


def concealed_hand_discard_win(pungs, kongs, chows, pair, drawn_or_discarded):
    pass


def all_chow_no_honors(pungs, kongs, pair, chows):
    pass


def tile_hog(pungs, kongs, pair, chows):
    pass


def mixed_double_pung(pungs, kongs):
    pass


def two_concealed_pungs(pungs, kongs):
    pass


def one_concealed_kong(kongs):
    pass


def all_simples(chows, pungs, kongs, pair):
    pass


def outside_hand(chows, pungs, kongs, pair):
    pass


def fully_concealed_self_drawn():
    raise NotImplemented("Do it in the score calc? Perhaps?")


def two_melded_kongs(kongs):
    pass


def last_tile():
    raise NotImplemented("Do it in the score calc")


def all_pungs(chows, kongs, pungs):
    pass


def half_flush(chows, pungs, kongs, pair):
    pass


def mixed_shifted_chow(chows):
    pass


def all_types():
    raise NotImplemented("Do it in the score calc")


def melded_hand():
    raise NotImplemented("Do it in the score calc")


def two_dragon_pungs(pungs, kongs):
    pass


def mixed_straight(chows):
    pass


def reversible_tiles(chows, pungs, kongs, pair):
    pass


def mixed_triple_chow(chows):
    pass


def mixed_shifted_pungs(pungs, kongs):
    pass


def two_concealed_kongs(kongs):
    pass


def last_tile_draw():
    raise NotImplemented("Do it in the score calc")


def last_tile_claim():
    raise NotImplemented("Do it in the score calc")


def replacement_win():
    raise NotImplemented("Do it in the score calc")


def kong_rob():
    raise NotImplemented("Do it in the score calc")


def chicken_hand():
    raise NotImplemented("Do it in the score calc after everything else is done. If pts = 0, then chicken hand.")


def upper_four(pungs, kongs, chows, pair):
    pass


def lower_four(pungs, kongs, chows, pair):
    pass


def big_three_winds(pungs, kongs):
    pass


def pure_straight(chows):
    pass


def three_suited_terminal_chows(chows):
    pass


def pure_shifted_chows(chows):
    pass


def all_fives(chows, pungs, kongs, pair):
    pass


def triple_pung(pungs, kongs):
    pass


def three_concealed_pungs(pungs, kongs):
    pass


def all_even_pungs(pungs, kongs):
    pass


def full_flush():
    raise NotImplemented("Do it in the score calc")


def pure_triple_chow(chows):
    pass


def pure_shifted_pungs(pungs, kongs):
    pass


def upper_tiles(chows, pungs, kongs, pair):
    pass


def middle_tiles(chows, pungs, kongs, pair):
    pass


def lower_tiles(chows, pungs, kongs, pair):
    pass


def four_shifted_chows(chows):
    pass


def three_kongs(kongs):
    pass


def all_terminals_and_honors():
    raise NotImplementedError("Do it in score calc.")


def quad_chow(chows):
    pass


def four_pure_shifted_pungs(pungs, kongs):
    pass


def all_terminals(chows, pungs, kongs, pair):
    pass


def all_honors():
    raise NotImplementedError("Do it in score calc.")


def little_four_winds(pungs, kongs):
    pass


def little_three_dragons(pungs, kongs):
    pass


def four_concealed_pungs(pungs, kongs):
    pass


def pure_terminal_chows(chows):
    pass


def big_four_winds(pungs, kongs):
    pass


def big_three_dragons(pungs, kongs):
    pass


def all_green(pungs, kongs, chows, pair):
    pass


def nine_gates(pungs, kongs, chows, pair):
    pass


def four_kongs(kongs):
    pass
