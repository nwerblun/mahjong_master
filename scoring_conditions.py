from game import *
from utilities import *
from functools import total_ordering


def lesser_honors_knitted_seq(hand):
    all_tiles = hand.concealed_tiles + [hand.drawn_tile] if hand.drawn_tile else hand.concealed_tiles
    if len(all_tiles) == 0 or len(hand.revealed_tiles) > 0 or len(all_tiles) < 14:
        return 0
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
    for variant in knitted_set_variants:
        uniques = set(all_tiles)
        knitted_uniques = set(variant + all_honor_tiles)
        if len(uniques.intersection(knitted_uniques)) >= 14:
            return 1
    return 0


def seven_pairs(hand):
    all_tiles = hand.concealed_tiles + [hand.drawn_tile] if hand.drawn_tile else hand.concealed_tiles
    if len(all_tiles) == 0 or len(hand.revealed_tiles) > 0 or len(all_tiles) < 14:
        return 0
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
    all_tiles = hand.concealed_tiles + [hand.drawn_tile] if hand.drawn_tile else hand.concealed_tiles
    if len(all_tiles) == 0 or len(hand.revealed_tiles) > 0 or len(all_tiles) < 14:
        return 0
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
    for variant in knitted_set_variants:
        uniques = set(all_tiles)
        honor_intersection = uniques.intersection(all_honor_tiles)
        knitted_uniques = set(all_honor_tiles + variant)
        if (len(honor_intersection) == len(all_honor_tiles)) and (len(uniques.intersection(knitted_uniques)) >= 14):
            return 1
    return 0


def seven_shifted_pairs(hand):
    if hand.get_num_honor_tiles() > 0:
        return 0
    all_tiles = hand.concealed_tiles + [hand.drawn_tile] if hand.drawn_tile else hand.concealed_tiles
    if len(all_tiles) == 0 or len(hand.revealed_tiles) > 0 or len(all_tiles) < 14:
        return 0
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
    if len(all_tiles) == 0 or len(hand.revealed_tiles) > 0 or len(all_tiles) < 14:
        return 0
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


@total_ordering
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
        elif set_type == "kong":
            self.numbers = tile_list[0].get_tile_number() + \
                           tile_list[1].get_tile_number() + \
                           tile_list[2].get_tile_number() + \
                           tile_list[3].get_tile_number()

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

    def __lt__(self, other):
        return self.numbers < other.numbers

    def __hash__(self):
        return (self.suit + self.numbers + self.set_type).__hash__()

    def __repr__(self):
        return self.suit + "_" + self.numbers


def pure_double_chow(chows):
    if not type(chows) == list or len(chows) == 0:
        return 0
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
        extra = pure_double_chow(recursion_list)
        amt += 1 + extra
    else:
        return 0
    return amt


def mixed_double_chow(chows):
    if not type(chows) == list or len(chows) == 0:
        return 0
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
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    if pair[0].get_tile_number() != "5":
        return 0
    for c in chows:
        if "5" not in c.numbers:
            return 0
    for p in pungs:
        if "5" not in p.numbers:
            return 0
    for k in kongs:
        if "5" not in k.numbers:
            return 0
    return 1


def triple_pung(pungs, kongs):
    fresh_pungs = [ts for ts in pungs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_kongs) + len(fresh_pungs) + len(used_but_not_excluded_kongs) \
        + len(used_but_not_excluded_pungs)
    if usable_sets < 3 or len(fresh_pungs) + len(fresh_kongs) == 0:
        return 0
    all_ts = sorted(flatten_list(fresh_pungs + used_but_not_excluded_pungs + fresh_kongs + used_but_not_excluded_kongs))
    numbers_list = [ts.numbers[0] for ts in all_ts]
    if sorted(numbers_list).count(all_ts[0].numbers[0]) == 3:
        start_ind = 0
    elif sorted(numbers_list).count(all_ts[1].numbers[0]) == 3:
        start_ind = 1
    else:
        return 0
    if not ((all_ts[start_ind+1].suit != all_ts[start_ind].suit) and
            (all_ts[start_ind+2].suit != all_ts[start_ind+1].suit)):
        return 0
    if not ((all_ts[start_ind+1].numbers[0] == all_ts[start_ind].numbers[0]) and
            (all_ts[start_ind+2].numbers[0] == all_ts[start_ind+1].numbers[0])):
        return 0
    for k in kongs:
        TileSet.update_used_excluded_stats(k)
    for p in pungs:
        TileSet.update_used_excluded_stats(p)
    return 1


def three_concealed_pungs(pungs, kongs):
    usable_sets = len(kongs) + len(pungs)
    if usable_sets < 3:
        return 0
    revealed_counter = 0
    for p in pungs:
        if not p.concealed:
            revealed_counter += 1
    for k in kongs:
        if not k.concealed:
            revealed_counter += 1
    if revealed_counter > 1:
        return 0
    return 1


def all_even_pungs(pungs, kongs, pair):
    usable_sets = len(pungs) + len(kongs)
    if usable_sets != 4:
        return 0
    if pair[0].is_dragon() or pair[0].is_wind():
        return 0
    if pair[0].get_tile_number() not in ["2", "4", "6", "8"]:
        return 0
    all_ts = flatten_list(pungs + kongs)
    evens = ["222", "2222", "444", "4444", "666", "6666", "888", "8888"]
    if not (
            (all_ts[0].numbers in evens) and
            (all_ts[1].numbers in evens) and
            (all_ts[2].numbers in evens) and
            (all_ts[3].numbers in evens)
    ):
        return 0
    return 1


def full_flush():
    raise NotImplemented("Do it in the score calc")


def pure_triple_chow(chows):
    fresh_chows = [ts for ts in chows if (not ts.used and not ts.excluded)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_chows) + len(used_but_not_excluded_chows)
    if usable_sets < 3 or len(fresh_chows) == 0:
        return 0
    if sorted(chows).count(chows[0]) == 3:
        start_ind = 0
    elif sorted(chows).count(chows[1]) == 3:
        start_ind = 1
    else:
        return 0
    if not ((chows[start_ind+2].suit == chows[start_ind+1].suit) and
            (chows[start_ind+1].suit == chows[start_ind].suit)):
        return 0
    for i in range(3):
        TileSet.update_used_excluded_stats(chows[start_ind+i])
    return 1


def pure_shifted_pungs(pungs, kongs):
    fresh_pungs = [ts for ts in pungs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_kongs) + len(fresh_pungs) + len(used_but_not_excluded_kongs) \
        + len(used_but_not_excluded_pungs)
    if usable_sets < 3 or len(fresh_pungs) + len(fresh_kongs) == 0:
        return 0
    all_ts = sorted(flatten_list(fresh_pungs + used_but_not_excluded_pungs + fresh_kongs + used_but_not_excluded_kongs))
    if not ((all_ts[1].suit == all_ts[0].suit) and (all_ts[2].suit == all_ts[1].suit)):
        return 0
    if not ((int(all_ts[1].numbers[0]) == int(all_ts[0].numbers[0]) + 1) and
            (int(all_ts[2].numbers[0]) == int(all_ts[1].numbers[0]) + 1)):
        return 0
    for k in kongs:
        TileSet.update_used_excluded_stats(k)
    for p in pungs:
        TileSet.update_used_excluded_stats(p)
    return 1


def upper_tiles(chows, pungs, kongs, pair):
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    if pair[0].get_tile_number() not in ["7", "8", "9"]:
        return 0
    for c in chows:
        if c.numbers != "789":
            return 0
    for p in pungs:
        if p.numbers not in ["777", "888", "999"]:
            return 0
    for k in kongs:
        if k.numbers not in ["7777", "8888", "9999"]:
            return 0
    return 1


def middle_tiles(chows, pungs, kongs, pair):
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    if pair[0].get_tile_number() not in ["4", "5", "6"]:
        return 0
    for c in chows:
        if c.numbers != "456":
            return 0
    for p in pungs:
        if p.numbers not in ["444", "555", "666"]:
            return 0
    for k in kongs:
        if k.numbers not in ["4444", "5555", "6666"]:
            return 0
    return 1


def lower_tiles(chows, pungs, kongs, pair):
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    if pair[0].get_tile_number() not in ["1", "2", "3"]:
        return 0
    for c in chows:
        if c.numbers != "123":
            return 0
    for p in pungs:
        if p.numbers not in ["111", "222", "333"]:
            return 0
    for k in kongs:
        if k.numbers not in ["1111", "2222", "3333"]:
            return 0
    return 1


def four_shifted_chows(chows):
    fresh_chows = [ts for ts in chows if (not ts.excluded and not ts.used)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    usable_chows = len(fresh_chows) + len(used_but_not_excluded_chows)
    if usable_chows != 4 or len(fresh_chows) == 0:
        return 0
    if not ((chows[0].suit == chows[1].suit) and (chows[2].suit == chows[1].suit) and (chows[3].suit == chows[2].suit)):
        return 0
    sorted_chows = sorted(chows)
    c0 = list(map(int, list(sorted_chows[0].numbers)))
    c1 = list(map(int, list(sorted_chows[1].numbers)))
    c2 = list(map(int, list(sorted_chows[2].numbers)))
    c3 = list(map(int, list(sorted_chows[3].numbers)))
    possibility_one = [
        [c1[i] - c0[i] for i in range(3)] == [1, 1, 1],
        [c2[i] - c1[i] for i in range(3)] == [1, 1, 1],
        [c3[i] - c2[i] for i in range(3)] == [1, 1, 1]
    ]
    possibility_two = [
        [c1[i] - c0[i] for i in range(3)] == [2, 2, 2],
        [c2[i] - c1[i] for i in range(3)] == [2, 2, 2],
        [c3[i] - c2[i] for i in range(3)] == [2, 2, 2]
    ]
    if all(possibility_one) or all(possibility_two):
        for c in chows:
            TileSet.update_used_excluded_stats(c)
        return 1
    return 0


def three_kongs(kongs):
    fresh_kongs = [ts for ts in kongs if (not ts.excluded and not ts.used)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    if len(fresh_kongs) < 1 or len(kongs) + len(used_but_not_excluded_kongs) < 3:
        return 0

    for k in fresh_kongs:
        TileSet.update_used_excluded_stats(k)
    for k in used_but_not_excluded_kongs:
        TileSet.update_used_excluded_stats(k)
    return 1


def all_terminals_and_honors(chows, pungs, kongs, pair):
    if not pair[0].is_wind() and not pair[0].is_dragon():
        if pair[0].get_tile_number() not in ["1", "9"]:
            return 0
    if len(chows) > 0:
        return 0
    for ts in pungs:
        if ts.numbers != "111" and ts.numbers != "999" and ts.suit != "dragon" and ts.suit != "wind":
            return 0
    for ts in kongs:
        if ts.numbers != "1111" and ts.numbers != "9999" and ts.suit != "dragon" and ts.suit != "wind":
            return 0
    return 1


def quad_chow(chows):
    fresh_chows = [ts for ts in chows if (not ts.used and not ts.excluded)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_chows) + len(used_but_not_excluded_chows)
    if usable_sets != 4 or len(fresh_chows) == 0:
        return 0
    if not (chows.count(chows[0])) == 4:
        return 0
    if not ((chows[0].suit == chows[1].suit) and (chows[2].suit == chows[1].suit) and (chows[3].suit == chows[2].suit)):
        return 0
    for c in chows:
        TileSet.update_used_excluded_stats(c)
    return 1


def four_pure_shifted_pungs(pungs, kongs):
    fresh_pungs = [ts for ts in pungs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_kongs) + len(fresh_pungs) + len(used_but_not_excluded_kongs) \
        + len(used_but_not_excluded_pungs)
    if usable_sets != 4 or len(fresh_pungs) + len(fresh_kongs) == 0:
        return 0
    all_ts = sorted(flatten_list(fresh_pungs + used_but_not_excluded_pungs + fresh_kongs + used_but_not_excluded_kongs))
    if not ((all_ts[1].suit == all_ts[0].suit) and (all_ts[2].suit == all_ts[1].suit)
            and (all_ts[3].suit == all_ts[2].suit)):
        return 0
    if not ((int(all_ts[1].numbers[0]) == int(all_ts[0].numbers[0]) + 1) and
            (int(all_ts[2].numbers[0]) == int(all_ts[1].numbers[0]) + 1) and
            (int(all_ts[3].numbers[0]) == int(all_ts[2].numbers[0]) + 1)):
        return 0
    for k in kongs:
        TileSet.update_used_excluded_stats(k)
    for p in pungs:
        TileSet.update_used_excluded_stats(p)
    return 1


def all_terminals(chows, pungs, kongs, pair):
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    if pair[0].get_tile_number() not in ["1", "9"]:
        return 0
    if len(chows) > 0:
        return 0
    for ts in pungs:
        if ts.numbers != "111" and ts.numbers != "999":
            return 0
    for ts in kongs:
        if ts.numbers != "1111" and ts.numbers != "9999":
            return 0
    return 1


def all_honors():
    raise NotImplementedError("Do it in score calc.")


def little_four_winds(pungs, kongs, pair):
    fresh_pungs = [ts for ts in pungs if (not ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    north_check, east_check, south_check, west_check = False, False, False, False
    fp_indicies_to_update = []
    up_indicies_to_update = []
    fk_indicies_to_update = []
    uk_indicies_to_update = []

    if not pair[0].is_wind():
        return 0

    if pair[0].get_wind_direction() == "e":
        east_check = True
    elif pair[0].get_wind_direction() == "n":
        north_check = True
    elif pair[0].get_wind_direction() == "w":
        west_check = True
    elif pair[0].get_wind_direction() == "s":
        south_check = True

    if not any([west_check, south_check, north_check, east_check]):
        return 0

    for i in range(len(fresh_pungs)):
        if fresh_pungs[i].suit == "wind":
            fp_indicies_to_update += [i]
            if fresh_pungs[i].numbers == "e" and not east_check:
                east_check = True
            elif fresh_pungs[i].numbers == "n" and not north_check:
                north_check = True
            elif fresh_pungs[i].numbers == "s" and not south_check:
                south_check = True
            elif fresh_pungs[i].numbers == "w" and not west_check:
                west_check = True

    for i in range(len(fresh_kongs)):
        if fresh_kongs[i].suit == "wind":
            fk_indicies_to_update += [i]
            if fresh_kongs[i].numbers == "e" and not east_check:
                east_check = True
            elif fresh_kongs[i].numbers == "s" and not south_check:
                south_check = True
            elif fresh_kongs[i].numbers == "n" and not north_check:
                north_check = True
            elif fresh_kongs[i].numbers == "w" and not west_check:
                west_check = True

    if not any([west_check, north_check, west_check, east_check]):
        # Must have at least one fresh set
        return 0

    for i in range(len(used_but_not_excluded_pungs)):
        if used_but_not_excluded_pungs[i].suit == "wind":
            up_indicies_to_update += [i]
            if used_but_not_excluded_pungs[i].numbers == "e" and not east_check:
                east_check = True
            elif used_but_not_excluded_pungs[i].numbers == "n" and not north_check:
                north_check = True
            elif used_but_not_excluded_pungs[i].numbers == "s" and not south_check:
                south_check = True
            elif used_but_not_excluded_pungs[i].numbers == "w" and not west_check:
                west_check = True

    if not all([west_check, east_check, north_check, south_check]):
        for i in range(len(used_but_not_excluded_kongs)):
            if used_but_not_excluded_kongs[i].suit == "wind":
                uk_indicies_to_update += [i]
                if used_but_not_excluded_kongs[i].numbers == "e" and not east_check:
                    east_check = True
                elif used_but_not_excluded_kongs[i].numbers == "s" and not south_check:
                    south_check = True
                elif used_but_not_excluded_kongs[i].numbers == "n" and not north_check:
                    north_check = True
                elif used_but_not_excluded_kongs[i].numbers == "w" and not west_check:
                    west_check = True

    if not all([west_check, east_check, south_check, north_check]):
        return 0
    for i in fp_indicies_to_update:
        TileSet.update_used_excluded_stats(fresh_pungs[i])
    for i in up_indicies_to_update:
        TileSet.update_used_excluded_stats(used_but_not_excluded_pungs[i])
    for i in fk_indicies_to_update:
        TileSet.update_used_excluded_stats(fresh_kongs[i])
    for i in uk_indicies_to_update:
        TileSet.update_used_excluded_stats(used_but_not_excluded_kongs[i])
    return 1


def little_three_dragons(pungs, kongs, pair):
    fresh_pungs = [ts for ts in pungs if (not ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    red_check, green_check, white_check = False, False, False
    fp_indicies_to_update = []
    up_indicies_to_update = []
    fk_indicies_to_update = []
    uk_indicies_to_update = []
    if not pair[0].is_dragon():
        return 0

    if pair[0].get_dragon_type() == "g":
        green_check = True
    elif pair[0].get_dragon_type() == "r":
        red_check = True
    elif pair[0].get_dragon_type() == "w":
        white_check = True

    if not any([green_check, white_check, red_check]):
        return 0

    for i in range(len(fresh_pungs)):
        if fresh_pungs[i].suit == "dragon":
            fp_indicies_to_update += [i]
            if fresh_pungs[i].numbers == "g" and not green_check:
                green_check = True
            elif fresh_pungs[i].numbers == "r" and not red_check:
                red_check = True
            elif fresh_pungs[i].numbers == "w" and not white_check:
                white_check = True

    for i in range(len(fresh_kongs)):
        if fresh_kongs[i].suit == "dragon":
            fk_indicies_to_update += [i]
            if fresh_kongs[i].numbers == "g" and not green_check:
                green_check = True
            elif fresh_kongs[i].numbers == "r" and not red_check:
                red_check = True
            elif fresh_kongs[i].numbers == "w" and not white_check:
                white_check = True

    if not any([red_check, green_check, white_check]):
        # Must have at least one unused set
        return 0

    for i in range(len(used_but_not_excluded_pungs)):
        if used_but_not_excluded_pungs[i].suit == "dragon":
            up_indicies_to_update += [i]
            if used_but_not_excluded_pungs[i].numbers == "g" and not green_check:
                green_check = True
            elif used_but_not_excluded_pungs[i].numbers == "r" and not red_check:
                red_check = True
            elif used_but_not_excluded_pungs[i].numbers == "w" and not white_check:
                white_check = True

    if not all([red_check, green_check, white_check]):
        for i in range(len(used_but_not_excluded_kongs)):
            if used_but_not_excluded_kongs[i].suit == "dragon":
                uk_indicies_to_update += [i]
                if used_but_not_excluded_kongs[i].numbers == "g" and not green_check:
                    green_check = True
                elif used_but_not_excluded_kongs[i].numbers == "r" and not red_check:
                    red_check = True
                elif used_but_not_excluded_kongs[i].numbers == "w" and not white_check:
                    white_check = True

    if not all([red_check, green_check, white_check]):
        return 0
    for i in fp_indicies_to_update:
        TileSet.update_used_excluded_stats(fresh_pungs[i])
    for i in up_indicies_to_update:
        TileSet.update_used_excluded_stats(used_but_not_excluded_pungs[i])
    for i in fk_indicies_to_update:
        TileSet.update_used_excluded_stats(fresh_kongs[i])
    for i in uk_indicies_to_update:
        TileSet.update_used_excluded_stats(used_but_not_excluded_kongs[i])
    return 1


def four_concealed_pungs(pungs, kongs):
    usable_sets = len(kongs) + len(pungs)
    if usable_sets != 4:
        return 0
    for p in pungs:
        if not p.concealed:
            return 0
    for k in kongs:
        if not k.concealed:
            return 0
    return 1


def pure_terminal_chows(chows, num_suits, num_honors, pair):
    fresh_chows = [ts for ts in chows if (not ts.used and not ts.excluded)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    if num_suits > 1 or num_honors > 0 or len(fresh_chows) == 0 or \
            (len(fresh_chows) + len(used_but_not_excluded_chows)) != 4:
        return 0
    if pair[0].get_tile_number() != "5":
        return 0
    lows_count = 0
    highs_count = 0
    for ts in fresh_chows:
        if ts.numbers == "123":
            lows_count += 1
        elif ts.numbers == "789":
            highs_count += 1

    for ts in used_but_not_excluded_chows:
        if ts.numbers == "123":
            lows_count += 1
        elif ts.numbers == "789":
            highs_count += 1

    if lows_count == 2 and highs_count == 2:
        for ts in chows:
            TileSet.update_used_excluded_stats(ts)
        return 1
    return 0


def big_four_winds(pungs, kongs):
    fresh_pungs = [ts for ts in pungs if (not ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    north_check, east_check, south_check, west_check = False, False, False, False
    fp_indicies_to_update = []
    up_indicies_to_update = []
    fk_indicies_to_update = []
    uk_indicies_to_update = []
    for i in range(len(fresh_pungs)):
        if fresh_pungs[i].suit == "wind":
            fp_indicies_to_update += [i]
            if fresh_pungs[i].numbers == "e":
                east_check = True
            elif fresh_pungs[i].numbers == "n":
                north_check = True
            elif fresh_pungs[i].numbers == "s":
                south_check = True
            elif fresh_pungs[i].numbers == "w":
                west_check = True

    for i in range(len(fresh_kongs)):
        if fresh_kongs[i].suit == "wind":
            fk_indicies_to_update += [i]
            if fresh_kongs[i].numbers == "e" and not east_check:
                east_check = True
            elif fresh_kongs[i].numbers == "s" and not south_check:
                south_check = True
            elif fresh_kongs[i].numbers == "n" and not north_check:
                north_check = True
            elif fresh_kongs[i].numbers == "w" and not west_check:
                west_check = True

    if not any([west_check, north_check, west_check, east_check]):
        # Must have at least one fresh set
        return 0

    for i in range(len(used_but_not_excluded_pungs)):
        if used_but_not_excluded_pungs[i].suit == "wind":
            up_indicies_to_update += [i]
            if used_but_not_excluded_pungs[i].numbers == "e" and not east_check:
                east_check = True
            elif used_but_not_excluded_pungs[i].numbers == "n" and not north_check:
                north_check = True
            elif used_but_not_excluded_pungs[i].numbers == "s" and not south_check:
                south_check = True
            elif used_but_not_excluded_pungs[i].numbers == "w" and not west_check:
                west_check = True

    if not all([west_check, east_check, north_check, south_check]):
        for i in range(len(used_but_not_excluded_kongs)):
            if used_but_not_excluded_kongs[i].suit == "wind":
                uk_indicies_to_update += [i]
                if used_but_not_excluded_kongs[i].numbers == "e" and not east_check:
                    east_check = True
                elif used_but_not_excluded_kongs[i].numbers == "s" and not south_check:
                    south_check = True
                elif used_but_not_excluded_kongs[i].numbers == "n" and not north_check:
                    north_check = True
                elif used_but_not_excluded_kongs[i].numbers == "w" and not west_check:
                    west_check = True

    if not all([west_check, east_check, south_check, north_check]):
        return 0
    for i in fp_indicies_to_update:
        TileSet.update_used_excluded_stats(fresh_pungs[i])
    for i in up_indicies_to_update:
        TileSet.update_used_excluded_stats(used_but_not_excluded_pungs[i])
    for i in fk_indicies_to_update:
        TileSet.update_used_excluded_stats(fresh_kongs[i])
    for i in uk_indicies_to_update:
        TileSet.update_used_excluded_stats(used_but_not_excluded_kongs[i])
    return 1


def big_three_dragons(pungs, kongs):
    fresh_pungs = [ts for ts in pungs if (not ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    red_check, green_check, white_check = False, False, False
    fp_indicies_to_update = []
    up_indicies_to_update = []
    fk_indicies_to_update = []
    uk_indicies_to_update = []
    for i in range(len(fresh_pungs)):
        if fresh_pungs[i].suit == "dragon":
            fp_indicies_to_update += [i]
            if fresh_pungs[i].numbers == "g":
                green_check = True
            elif fresh_pungs[i].numbers == "r":
                red_check = True
            elif fresh_pungs[i].numbers == "w":
                white_check = True

    for i in range(len(fresh_kongs)):
        if fresh_kongs[i].suit == "dragon":
            fk_indicies_to_update += [i]
            if fresh_kongs[i].numbers == "g" and not green_check:
                green_check = True
            elif fresh_kongs[i].numbers == "r" and not red_check:
                red_check = True
            elif fresh_kongs[i].numbers == "w" and not white_check:
                white_check = True

    if not any([red_check, green_check, white_check]):
        # Must have at least one unused set
        return 0

    for i in range(len(used_but_not_excluded_pungs)):
        if used_but_not_excluded_pungs[i].suit == "dragon":
            up_indicies_to_update += [i]
            if used_but_not_excluded_pungs[i].numbers == "g" and not green_check:
                green_check = True
            elif used_but_not_excluded_pungs[i].numbers == "r" and not red_check:
                red_check = True
            elif used_but_not_excluded_pungs[i].numbers == "w" and not white_check:
                white_check = True

    if not all([red_check, green_check, white_check]):
        for i in range(len(used_but_not_excluded_kongs)):
            if used_but_not_excluded_kongs[i].suit == "dragon":
                uk_indicies_to_update += [i]
                if used_but_not_excluded_kongs[i].numbers == "g" and not green_check:
                    green_check = True
                elif used_but_not_excluded_kongs[i].numbers == "r" and not red_check:
                    red_check = True
                elif used_but_not_excluded_kongs[i].numbers == "w" and not white_check:
                    white_check = True

    if not all([red_check, green_check, white_check]):
        return 0
    for i in fp_indicies_to_update:
        TileSet.update_used_excluded_stats(fresh_pungs[i])
    for i in up_indicies_to_update:
        TileSet.update_used_excluded_stats(used_but_not_excluded_pungs[i])
    for i in fk_indicies_to_update:
        TileSet.update_used_excluded_stats(fresh_kongs[i])
    for i in uk_indicies_to_update:
        TileSet.update_used_excluded_stats(used_but_not_excluded_kongs[i])
    return 1


def all_green(pungs, kongs, chows, pair):
    all_tilesets_list = flatten_list([pungs, kongs, chows])
    if pair[0].is_dragon() and not (pair[0].get_dragon_type() == "g"):
        return 0
    elif pair[0].is_bamboo() and pair[0].get_tile_number() not in ["2", "3", "4", "6", "8"]:
        return 0

    for ts in all_tilesets_list:
        if ts.suit == "dragon" and not (ts.numbers == "g"):
            return 0
        elif ts.suit == "bamboo" and ts.numbers not in \
                ["222", "234", "333", "444", "666", "888", "2222", "3333", "4444", "6666", "8888"]:
            return 0
        elif ts.suit != "bamboo":
            return 0
    return 1


def nine_gates(pungs, kongs, chows, pair, num_suits, num_honors):
    if num_suits > 1 or num_honors > 0 or len(kongs) > 0 or len(chows) < 2 or len(pungs) == 0:
        return 0
    for p in pungs:
        if (not p.concealed) or p.declared:
            return 0
    for c in chows:
        if (not c.concealed) or c.declared:
            return 0
    if pair[0].get_tile_number() == "1":
        # 11 123 345 678 999
        # 11 123 456 678 999
        # 11 123 456 789 999
        if pungs[0].numbers == "999" and chows[0].numbers == "123":
            if chows[1].numbers == "345" and chows[2].numbers == "678":
                return 1
            elif chows[1].numbers == "456" and (chows[2].numbers == "678" or chows[2].numbers == "789"):
                return 1
        else:
            return 0
    elif pair[0].get_tile_number() == "9":
        # 111 123 456 789 99
        # 111 234 456 789 99
        # 111 234 567 789 99
        if pungs[0].numbers == "111" and chows[2].numbers == "789":
            if chows[0].numbers == "123" and chows[1].numbers == "456":
                return 1
            elif chows[0].numbers == "234" and (chows[2].numbers == "456" or chows[2].numbers == "567"):
                return 1
        else:
            return 0
    elif pair[0].get_tile_number() == "2":
        # 111 22 345 678 999
        if pungs[0].numbers == "111" and pungs[1].numbers == "999" and\
                chows[0].numbers == "345" and chows[1].numbers == "678":
            return 1
        else:
            return 0
    elif pair[0].get_tile_number() == "5":
        # 111 234 55 678 999
        if pungs[0].numbers == "111" and pungs[1].numbers == "999" and \
                chows[0].numbers == "234" and chows[1].numbers == "678":
            return 1
        else:
            return 0
    elif pair[0].get_tile_number() == "8":
        # 111 234 567 88 999
        if pungs[0].numbers == "111" and pungs[1].numbers == "999" and \
                chows[0].numbers == "234" and chows[1].numbers == "567":
            return 1
        else:
            return 0
    else:
        return 0


def four_kongs(kongs):
    fresh_kongs = [ts for ts in kongs if (not ts.excluded and not ts.used)]
    excluded_kongs = [ts for ts in kongs if ts.excluded]
    if len(fresh_kongs) < 1 or len(excluded_kongs) > 0 or len(kongs) != 4:
        return 0

    for k in kongs:
        TileSet.update_used_excluded_stats(k)
    return 1
