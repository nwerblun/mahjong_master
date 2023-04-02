from game import *
from utilities import *
from functools import total_ordering


def lesser_honors_knitted_seq(hand):
    all_tiles = hand.concealed_tiles
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
            return 12
    return 0


def seven_pairs(hand):
    all_tiles = hand.concealed_tiles
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
    return 24


def greater_honors_knitted_tiles(hand):
    all_tiles = hand.concealed_tiles
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
            return 24
    return 0


def seven_shifted_pairs(hand):
    if hand.get_num_honor_tiles() > 0:
        return 0
    all_tiles = hand.concealed_tiles
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
    return 88


def thirteen_orphans(hand):
    all_tiles = hand.concealed_tiles
    if len(all_tiles) == 0 or len(hand.revealed_tiles) > 0 or len(all_tiles) < 14:
        return 0
    hand_size = len(hand.concealed_tiles)
    tiles_needed = [
        Tile("b1"), Tile("b9"), Tile("c1"),
        Tile("c9"), Tile("d1"), Tile("d9"),
        Tile("drg"), Tile("drr"), Tile("drw"),
        Tile("we"), Tile("wn"), Tile("ws"),
        Tile("ww")
    ]
    duplicate_tile_possibilities = tiles_needed[:]
    ctr = 0
    dupe_found = False
    while ctr <= 13:
        if all_tiles[0] in tiles_needed:
            tiles_needed.pop(tiles_needed.index(all_tiles[0]))
            all_tiles.pop(0)
        elif all_tiles[0] in duplicate_tile_possibilities:
            dupe_found = True
            all_tiles.pop(0)
        else:
            all_tiles.pop(0)
        ctr += 1

    if len(tiles_needed) > 0:
        return 0
    if hand_size >= 14 and dupe_found:
        return 88
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
        return pure_double_chow(fresh_chows[1:] + used_but_not_excluded_chows)
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
        return mixed_double_chow(fresh_chows[1:] + used_but_not_excluded_chows)
    return amt


def short_straight(chows):
    s_chows = sorted([ts for ts in chows if not ts.excluded])
    if len(s_chows) < 2:
        return 0
    amt = 0
    for i in range(1, len(s_chows)):
        if (s_chows[i].suit == s_chows[i-1].suit) and (int(s_chows[i].numbers[0]) == int(s_chows[i-1].numbers[0]) + 3):
            if (not s_chows[i].used and not s_chows[i].excluded) or\
                    (not s_chows[i-1].used and not s_chows[i-1].excluded):
                amt += 1
                TileSet.update_used_excluded_stats([s_chows[i], s_chows[i-1]])
    return amt


def two_terminal_chows(chows):
    fresh_chows = [ts for ts in chows if (not ts.excluded and not ts.used)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    low_terms_fresh = [ts for ts in fresh_chows if ts.numbers == "123"]
    high_terms_fresh = [ts for ts in fresh_chows if ts.numbers == "789"]
    low_terms_used = [ts for ts in used_but_not_excluded_chows if ts.numbers == "123"]
    high_terms_used = [ts for ts in used_but_not_excluded_chows if ts.numbers == "789"]
    if len(low_terms_fresh) == 0 or len(high_terms_fresh) == 0:
        return 0
    low_terms_fresh = sorted(low_terms_fresh, key=lambda x: x.suit)
    high_terms_fresh = sorted(high_terms_fresh, key=lambda x: x.suit)
    low_terms_used = sorted(low_terms_used, key=lambda x: x.suit)
    high_terms_used = sorted(high_terms_used, key=lambda x: x.suit)
    low_f_h_u = list(zip(low_terms_fresh, high_terms_used))
    low_u_h_f = list(zip(low_terms_used, high_terms_fresh))
    low_f_h_f = list(zip(low_terms_fresh, high_terms_fresh))
    amt = 0
    for i in range(len(low_f_h_u)):
        if low_f_h_u[i][0].suit == low_f_h_u[i][1].suit:
            amt += 1
            TileSet.update_used_excluded_stats([low_f_h_u[i][0], low_f_h_u[i][1]])
    for i in range(len(low_u_h_f)):
        if low_u_h_f[i][0].suit == low_u_h_f[i][1].suit:
            amt += 1
            TileSet.update_used_excluded_stats([low_u_h_f[i][0], low_u_h_f[i][1]])
    for i in range(len(low_f_h_f)):
        if low_f_h_f[i][0].suit == low_f_h_f[i][1].suit:
            # Possibly updated from previous loops
            if (low_f_h_f[i][0].used or low_f_h_f[i][0].excluded) and\
                    (low_f_h_f[i][1].used or low_f_h_f[i][1].excluded):
                continue
            amt += 1
            TileSet.update_used_excluded_stats([low_f_h_f[i][0], low_f_h_f[i][1]])
    if amt == 0:
        # No pairs possible, stop here
        return 0
    amt += two_terminal_chows(chows)
    return amt


def terminal_non_dragon_honor_pung(pungs, kongs, seat_wind, round_wind):
    # Apparently not victim to the "used" rule.
    amt = 0
    used_p_indicies = []
    used_k_indicies = []
    for i in range(len(pungs)):
        if pungs[i].suit == "wind" and pungs[i].numbers not in [round_wind, seat_wind]:
            amt += 1
        elif pungs[i].suit != "dragon" and pungs[i].numbers in ["111", "999"]:
            amt += 1
    for i in range(len(kongs)):
        if kongs[i].suit == "wind" and kongs[i].numbers not in [round_wind, seat_wind]:
            amt += 1
        elif kongs[i].suit != "dragon" and kongs[i].numbers in ["1111", "9999"]:
            amt += 1
    return amt


def melded_kong(kongs):
    if len(kongs) == 0:
        return 0
    amt = 0
    for k in kongs:
        if not k.concealed:
            amt += 1
    return amt


def voided_suit():
    raise NotImplemented("Do it in the score calc")


def no_honor_tiles():
    raise NotImplemented("Do it in the score calc")


def self_drawn():
    raise NotImplemented("Do it in the score calc? Maybe?")


def flowers():
    return 0


def edge_wait(chows):
    raise NotImplementedError("Only considering winning hands at the moment, can't check pre-win hands.")


def closed_wait(chows):
    raise NotImplementedError("Only considering winning hands at the moment, can't check pre-win hands.")


def single_wait(pair):
    raise NotImplementedError("Only considering winning hands at the moment, can't check pre-win hands.")


def dragon_pung(pungs, kongs):
    # Apparently doesn't fall victim to the "used" rule
    amt = 0
    for i in range(len(pungs)):
        if pungs[i].suit == "dragon":
            amt += 1
    for i in range(len(kongs)):
        if kongs[i].suit == "dragon":
            amt += 1
    return amt


def round_wind_pung(pungs, kongs, round_wind):
    return seat_wind_pung(pungs, kongs, round_wind)


def seat_wind_pung(pungs, kongs, seat_wind):
    for i in range(len(pungs)):
        if pungs[i].suit == "wind" and pungs[i].numbers == seat_wind:
            return 1
    for i in range(len(kongs)):
        if kongs[i].suit == "wind" and kongs[i].numbers == seat_wind:
            return 1
    return 0


def concealed_hand_discard_win():
    raise NotImplemented("Do it in the score calc? Maybe?")


def all_chow_no_honors(pungs, kongs, pair):
    if len(kongs) != 0 or len(pungs) != 0:
        return 0
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    return 1


def tile_hog(pungs, kongs, pair, chows):
    all_bamboo = [ts for ts in pungs+kongs+chows if ts.suit == "bamboo"]
    all_dots = [ts for ts in pungs+kongs+chows if ts.suit == "dot"]
    all_chars = [ts for ts in pungs+kongs+chows if ts.suit == "character"]

    all_bamboo_nums = flatten_list(list(map(lambda x: list(x.numbers), all_bamboo)))
    all_dot_nums = flatten_list(list(map(lambda x: list(x.numbers), all_dots)))
    all_char_nums = flatten_list(list(map(lambda x: list(x.numbers), all_chars)))
    pair_num = 0
    if not pair[0].is_wind() and not pair[0].is_dragon():
        pair_num = pair[0].get_tile_number()
        if pair[0].is_bamboo():
            all_bamboo_nums += [pair_num]*2
        elif pair[0].is_dot():
            all_dot_nums += [pair_num]*2
        elif pair[0].is_char():
            all_char_nums += [pair_num]*2
    amt = 0
    include = True
    for i in range(1, 10):
        if all_bamboo_nums.count(str(i)) == 4:
            for k in [ts for ts in all_bamboo if ts.set_type == "kong"]:
                if str(i) in k.numbers:
                    include = False
            if include:
                amt += 1
        include = True
        if all_dot_nums.count(str(i)) == 4:
            for k in [ts for ts in all_dots if ts.set_type == "kong"]:
                if str(i) in k.numbers:
                    include = False
            if include:
                amt += 1
        include = True
        if all_char_nums.count(str(i)) == 4:
            for k in [ts for ts in all_chars if ts.set_type == "kong"]:
                if str(i) in k.numbers:
                    include = False
            if include:
                amt += 1
        include = True
    return amt


def mixed_double_pung(pungs, kongs):
    fresh_pungs = [ts for ts in pungs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_kongs) + len(fresh_pungs) + len(used_but_not_excluded_kongs) \
        + len(used_but_not_excluded_pungs)
    if usable_sets < 2 or len(fresh_pungs) + len(fresh_kongs) == 0:
        return 0
    all_ts = sorted(fresh_pungs + used_but_not_excluded_pungs + fresh_kongs + used_but_not_excluded_kongs)
    numbers_list = [ts.numbers[0] for ts in all_ts]
    if numbers_list.count(all_ts[0].numbers[0]) >= 2 and (not all_ts[1].used or not all_ts[0].used):
        start_ind = 0
    elif numbers_list.count(all_ts[1].numbers[0]) >= 2 and (not all_ts[2].used or not all_ts[1].used):
        start_ind = 1
    elif len(numbers_list) > 2 and numbers_list.count(all_ts[2].numbers[0]) >= 2:
        start_ind = 2
    else:
        return 0
    if all_ts[start_ind + 1].suit == all_ts[start_ind].suit:
        return 0
    if all_ts[start_ind + 1].numbers[0] != all_ts[start_ind].numbers[0]:
        return 0
    TileSet.update_used_excluded_stats([all_ts[start_ind], all_ts[start_ind+1]])
    return 1


def two_concealed_pungs(pungs, kongs):
    # Don't check kongs since there's a two concealed kongs already
    concealed = [ts for ts in pungs if ts.concealed]
    usable_kongs = [ts for ts in kongs if not ts.used and not ts.excluded]
    if len(concealed) >= 2:
        return 1
    elif len(concealed) == 1 and len(usable_kongs) >= 1:
        return 1
    return 0


def one_concealed_kong(kongs):
    if len(kongs) == 0:
        return 0
    amt = 0
    for k in kongs:
        if k.concealed:
            amt += 1
    return amt


def all_simples(chows, pungs, kongs, pair):
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    elif pair[0].get_tile_number() in ["1", "9"]:
        return 0
    for ts in (chows + pungs + kongs):
        if "1" in ts.numbers or "9" in ts.numbers:
            return 0
        elif ts.suit == "dragon" or ts.suit == "wind":
            return 0
    return 1


def outside_hand(chows, pungs, kongs, pair):
    if (not pair[0].is_wind() and not pair[0].is_dragon()) and (pair[0].get_tile_number() not in ["1", "9"]):
        return 0
    for ts in (chows + pungs + kongs):
        if (not ts.suit == "dragon" and not ts.suit == "wind") and ("1" not in ts.numbers and "9" not in ts.numbers):
            return 0
    return 1


def fully_concealed_self_drawn():
    raise NotImplemented("Do it in the score calc? Perhaps?")


def two_melded_kongs(kongs):
    # You get 6 points for melded + unmelded. Maybe ignore concealed status here, so you get 4 + 2 pts?
    # Apparently doesn't combine with 3 kongs, but 1 concealed kong does??
    if len(kongs) != 2:
        return 0
    for k in kongs:
        if not k.concealed:
            return 1
    return 0


def last_tile():
    raise NotImplemented("Do it in the score calc")


def all_pungs(chows, kongs, pungs):
    raise NotImplemented("Do it in the score calc")


def half_flush(chows, pungs, kongs, pair):
    raise NotImplemented("Do it in the score calc")


def mixed_shifted_chow(chows):
    fresh_chows = [ts for ts in chows if (not ts.used and not ts.excluded)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_chows) + len(used_but_not_excluded_chows)
    if usable_sets < 3 or len(fresh_chows) == 0:
        return 0
    suit_one_check, suit_two_check, suit_three_check = False, False, False
    sorted_used = sorted(used_but_not_excluded_chows)
    to_be_used = []
    for p in sorted_used:
        if p.suit != "wind" and p.suit != "dragon":
            if not suit_one_check:
                suit_one_check = True
                to_be_used += [p]
            elif suit_one_check and not suit_two_check and p.suit != to_be_used[0].suit and \
                    (int(p.numbers[0]) == int(to_be_used[0].numbers[0]) + 1):
                suit_two_check = True
                to_be_used += [p]
            elif suit_one_check and suit_two_check and not suit_three_check and p.suit != to_be_used[0].suit \
                    and p.suit != to_be_used[1].suit and \
                    (int(p.numbers[0]) == int(to_be_used[1].numbers[0]) + 1):
                suit_three_check = True
                to_be_used += [p]

    sorted_fresh = sorted(fresh_chows)
    for p in sorted_fresh:
        if p.suit != "wind" and p.suit != "dragon":
            if not suit_one_check:
                suit_one_check = True
                to_be_used += [p]
            elif suit_one_check and not suit_two_check and p.suit != to_be_used[0].suit and \
                    (int(p.numbers[0]) == int(to_be_used[0].numbers[0]) + 1):
                suit_two_check = True
                to_be_used += [p]
            elif suit_one_check and suit_two_check and not suit_three_check and p.suit != to_be_used[0].suit \
                    and p.suit != to_be_used[1].suit and \
                    (int(p.numbers[0]) == int(to_be_used[1].numbers[0]) + 1):
                suit_three_check = True
                to_be_used += [p]

    if not all([suit_three_check, suit_two_check, suit_one_check]):
        return 0
    TileSet.update_used_excluded_stats(to_be_used)
    return 1


def all_types():
    raise NotImplemented("Do it in the score calc")


def melded_hand():
    raise NotImplemented("Do it in the score calc")


def two_dragon_pungs(pungs, kongs):
    fresh_pungs = [ts for ts in pungs if (not ts.excluded and not ts.used)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.excluded and not ts.used)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_kongs) + len(fresh_pungs) + len(used_but_not_excluded_kongs) \
        + len(used_but_not_excluded_pungs)
    if usable_sets < 2 or len(fresh_pungs) + len(fresh_kongs) == 0:
        return 0
    first = None
    second = None
    fresh_dragons = [ts for ts in fresh_pungs+fresh_kongs if ts.suit == "dragon"]
    used_dragons = [ts for ts in used_but_not_excluded_pungs+used_but_not_excluded_kongs if ts.suit == "dragon"]
    if len(used_dragons) > 0:
        first = (used_but_not_excluded_pungs + used_but_not_excluded_kongs)[0]
        second = (fresh_pungs + fresh_kongs)[0]
    elif len(fresh_dragons) > 1:
        first = (fresh_pungs + fresh_kongs)[0]
        second = (fresh_pungs + fresh_kongs)[1]
    else:
        return 0
    TileSet.update_used_excluded_stats([first, second])
    return 1


def mixed_straight(chows):
    fresh_chows = [ts for ts in chows if (not ts.used and not ts.excluded)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_chows) + len(used_but_not_excluded_chows)
    if usable_sets < 3 or len(fresh_chows) == 0:
        return 0
    suit_one_check, suit_two_check, suit_three_check = False, False, False
    sorted_used = sorted(used_but_not_excluded_chows)
    to_be_used = []
    for p in sorted_used:
        if p.suit != "wind" and p.suit != "dragon":
            if not suit_one_check:
                suit_one_check = True
                to_be_used += [p]
            elif suit_one_check and not suit_two_check and p.suit != to_be_used[0].suit and \
                    (int(p.numbers[0]) == int(to_be_used[0].numbers[0]) + 3):
                suit_two_check = True
                to_be_used += [p]
            elif suit_one_check and suit_two_check and not suit_three_check and p.suit != to_be_used[0].suit \
                    and p.suit != to_be_used[1].suit and \
                    (int(p.numbers[0]) == int(to_be_used[1].numbers[0]) + 3):
                suit_three_check = True
                to_be_used += [p]

    sorted_fresh = sorted(fresh_chows)
    for p in sorted_fresh:
        if p.suit != "wind" and p.suit != "dragon":
            if not suit_one_check:
                suit_one_check = True
                to_be_used += [p]
            elif suit_one_check and not suit_two_check and p.suit != to_be_used[0].suit and \
                    (int(p.numbers[0]) == int(to_be_used[0].numbers[0]) + 3):
                suit_two_check = True
                to_be_used += [p]
            elif suit_one_check and suit_two_check and not suit_three_check and p.suit != to_be_used[0].suit \
                    and p.suit != to_be_used[1].suit and \
                    (int(p.numbers[0]) == int(to_be_used[1].numbers[0]) + 3):
                suit_three_check = True
                to_be_used += [p]

    if not all([suit_three_check, suit_two_check, suit_one_check]):
        return 0
    TileSet.update_used_excluded_stats(to_be_used)
    return 1


def reversible_tiles(chows, pungs, kongs, pair):
    all_tilesets_list = flatten_list([pungs, kongs, chows])
    if pair[0].is_dragon() and not (pair[0].get_dragon_type() == "w"):
        return 0
    elif pair[0].is_char():
        return 0
    elif pair[0].is_bamboo() and pair[0].get_tile_number() not in ["2", "4", "5", "6", "8", "9"]:
        return 0
    elif pair[0].is_dot() and pair[0].get_tile_number() not in ["1", "2", "3", "4", "5", "8", "9"]:
        return 0

    for ts in all_tilesets_list:
        if ts.suit == "dragon" and not (ts.numbers == "w"):
            return 0
        elif ts.suit == "bamboo" and ts.numbers not in \
                ["222", "444", "456", "555", "666", "888", "999", "2222", "4444", "5555", "6666", "8888", "9999"]:
            return 0
        elif ts.suit == "dot" and ts.numbers not in ["111", "123", "222", "234", "333", "345", "444", "555", "888",
                                                     "999", "1111", "2222", "3333", "4444", "5555", "8888", "9999"]:
            return 0
        elif ts.suit == "character":
            return 0
    return 1


def mixed_triple_chow(chows):
    fresh_chows = [ts for ts in chows if (not ts.used and not ts.excluded)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_chows) + len(used_but_not_excluded_chows)
    if usable_sets < 3 or len(fresh_chows) == 0:
        return 0
    suit_one_check, suit_two_check, suit_three_check = False, False, False
    sorted_used = sorted(used_but_not_excluded_chows)
    to_be_used = []
    for p in sorted_used:
        if p.suit != "wind" and p.suit != "dragon":
            if not suit_one_check:
                suit_one_check = True
                to_be_used += [p]
            elif suit_one_check and not suit_two_check and p.suit != to_be_used[0].suit and \
                    (int(p.numbers[0]) == int(to_be_used[0].numbers[0])):
                suit_two_check = True
                to_be_used += [p]
            elif suit_one_check and suit_two_check and not suit_three_check and p.suit != to_be_used[0].suit \
                    and p.suit != to_be_used[1].suit and \
                    (int(p.numbers[0]) == int(to_be_used[1].numbers[0])):
                suit_three_check = True
                to_be_used += [p]

    sorted_fresh = sorted(fresh_chows)
    for p in sorted_fresh:
        if p.suit != "wind" and p.suit != "dragon":
            if not suit_one_check:
                suit_one_check = True
                to_be_used += [p]
            elif suit_one_check and not suit_two_check and p.suit != to_be_used[0].suit and \
                    (int(p.numbers[0]) == int(to_be_used[0].numbers[0])):
                suit_two_check = True
                to_be_used += [p]
            elif suit_one_check and suit_two_check and not suit_three_check and p.suit != to_be_used[0].suit \
                    and p.suit != to_be_used[1].suit and \
                    (int(p.numbers[0]) == int(to_be_used[1].numbers[0])):
                suit_three_check = True
                to_be_used += [p]

    if not all([suit_three_check, suit_two_check, suit_one_check]):
        return 0
    TileSet.update_used_excluded_stats(to_be_used)
    return 1


def mixed_shifted_pungs(pungs, kongs):
    fresh_pungs = [ts for ts in pungs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_pungs = [ts for ts in pungs if (ts.used and not ts.excluded)]
    fresh_kongs = [ts for ts in kongs if (not ts.used and not ts.excluded)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    usable_sets = len(fresh_kongs) + len(fresh_pungs) + len(used_but_not_excluded_kongs) \
        + len(used_but_not_excluded_pungs)
    if usable_sets < 3 or len(fresh_pungs) + len(fresh_kongs) == 0:
        return 0
    suit_one_check, suit_two_check, suit_three_check = False, False, False
    sorted_used = sorted(used_but_not_excluded_kongs + used_but_not_excluded_pungs)
    to_be_used = []
    for p in sorted_used:
        if p.suit != "wind" and p.suit != "dragon":
            if not suit_one_check:
                suit_one_check = True
                to_be_used += [p]
            elif suit_one_check and not suit_two_check and p.suit != to_be_used[0].suit and \
                    (int(p.numbers[0]) == (int(to_be_used[0].numbers[0]) + 1)):
                suit_two_check = True
                to_be_used += [p]
            elif suit_one_check and suit_two_check and not suit_three_check and p.suit != to_be_used[0].suit\
                    and p.suit != to_be_used[1].suit and \
                    (int(p.numbers[0]) == (int(to_be_used[1].numbers[0]) + 1)):
                suit_three_check = True
                to_be_used += [p]

    sorted_fresh = sorted(fresh_pungs + fresh_kongs)
    for p in sorted_fresh:
        if p.suit != "wind" and p.suit != "dragon":
            if not suit_one_check:
                suit_one_check = True
                to_be_used += [p]
            elif suit_one_check and not suit_two_check and p.suit != to_be_used[0].suit and \
                    (int(p.numbers[0]) == (int(to_be_used[0].numbers[0]) + 1)):
                suit_two_check = True
                to_be_used += [p]
            elif suit_one_check and suit_two_check and not suit_three_check and p.suit != to_be_used[0].suit\
                    and p.suit != to_be_used[1].suit and \
                    (int(p.numbers[0]) == (int(to_be_used[1].numbers[0]) + 1)):
                suit_three_check = True
                to_be_used += [p]

    if not all([suit_three_check, suit_two_check, suit_one_check]):
        return 0
    TileSet.update_used_excluded_stats(to_be_used)
    return 1


def two_concealed_kongs(kongs):
    ckongs = [ts for ts in kongs if ts.concealed]
    if len(ckongs) < 2:
        return 0
    return 1


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
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    if pair[0].get_tile_number() not in ["6", "7", "8", "9"]:
        return 0
    for c in chows:
        if c.numbers != "678" and c.numbers != "789":
            return 0
    for p in pungs:
        if p.numbers not in ["666", "777", "888", "999"]:
            return 0
    for k in kongs:
        if k.numbers not in ["6666", "7777", "8888", "9999"]:
            return 0
    return 1


def lower_four(pungs, kongs, chows, pair):
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    if pair[0].get_tile_number() not in ["1", "2", "3", "4"]:
        return 0
    for c in chows:
        if c.numbers != "123" and c.numbers != "234":
            return 0
    for p in pungs:
        if p.numbers not in ["111", "222", "333", "444"]:
            return 0
    for k in kongs:
        if k.numbers not in ["1111", "2222", "3333", "4444"]:
            return 0
    return 1


def big_three_winds(pungs, kongs):
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

    if not ([west_check, east_check, south_check, north_check].count(True) >= 3):
        return 0
    TileSet.update_used_excluded_stats([fresh_pungs[i] for i in fp_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_pungs[i] for i in up_indicies_to_update])
    TileSet.update_used_excluded_stats([fresh_kongs[i] for i in fk_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_kongs[i] for i in uk_indicies_to_update])
    return 1


def pure_straight(chows):
    fresh_chows = [ts for ts in chows if (not ts.excluded and not ts.used)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    usable_chows = len(fresh_chows) + len(used_but_not_excluded_chows)
    if usable_chows < 3 or len(fresh_chows) == 0:
        return 0
    suit_map = sorted(list(map(lambda x: x.suit, fresh_chows + used_but_not_excluded_chows)))
    most_suit = suit_map[0] if suit_map.count(suit_map[0]) >= 3 else suit_map[1]

    to_be_used = []
    for ts in fresh_chows:
        if ts.suit == most_suit:
            to_be_used += [ts]

    if len(to_be_used) == 0:
        return 0

    for ts in used_but_not_excluded_chows:
        if ts.suit == most_suit:
            to_be_used += [ts]

    if len(to_be_used) < 3:
        return 0

    sorted_chows = sorted(to_be_used)
    if sorted_chows[0].numbers == "123" and sorted_chows[1].numbers == "456" and sorted_chows[2].numbers == "789":
        TileSet.update_used_excluded_stats(sorted_chows)
        return 1
    return 0


def three_suited_terminal_chows(chows, pair):
    fresh_chows = [ts for ts in chows if (not ts.excluded and not ts.used)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    usable_chows = len(fresh_chows) + len(used_but_not_excluded_chows)
    if usable_chows != 4 or len(fresh_chows) == 0:
        return 0
    if pair[0].is_wind() or pair[0].is_dragon():
        return 0
    if pair[0].get_tile_number() != "5":
        return 0
    pair_suit = pair[0].type
    suit_one = ""
    low_suit_one, high_suit_one = False, False
    low_suit_two, high_suit_two = False, False
    s_chows = sorted(chows)
    for c in s_chows:
        if (c.numbers == "123") and (not low_suit_one) and (c.suit != pair_suit):
            suit_one = c.suit
            low_suit_one = True
        elif (c.numbers == "789") and (not high_suit_one) and (c.suit == suit_one) and (c.suit != pair_suit):
            high_suit_one = True
        elif (c.numbers == "123") and (not low_suit_two) and (c.suit != suit_one) and (c.suit != pair_suit):
            low_suit_two = True
        elif (c.numbers == "789") and (not high_suit_two) and (c.suit != suit_one) and (c.suit != pair_suit):
            high_suit_two = True

    if all([low_suit_one, low_suit_two, high_suit_one, high_suit_two]):
        TileSet.update_used_excluded_stats(chows)
        return 1
    return 0


def pure_shifted_chows(chows):
    fresh_chows = [ts for ts in chows if (not ts.excluded and not ts.used)]
    used_but_not_excluded_chows = [ts for ts in chows if (ts.used and not ts.excluded)]
    usable_chows = len(fresh_chows) + len(used_but_not_excluded_chows)
    if usable_chows < 3 or len(fresh_chows) == 0:
        return 0
    suit_map = sorted(list(map(lambda x: x.suit, fresh_chows + used_but_not_excluded_chows)))
    most_suit = suit_map[0] if suit_map.count(suit_map[0]) >= 3 else suit_map[1]

    to_be_used = []
    for ts in fresh_chows:
        if ts.suit == most_suit:
            to_be_used += [ts]

    if len(to_be_used) == 0:
        return 0

    for ts in used_but_not_excluded_chows:
        if ts.suit == most_suit:
            to_be_used += [ts]

    if len(to_be_used) < 3:
        return 0

    sorted_chows = sorted(to_be_used)
    c0 = list(map(int, list(sorted_chows[0].numbers)))
    c1 = list(map(int, list(sorted_chows[1].numbers)))
    c2 = list(map(int, list(sorted_chows[2].numbers)))
    possibility_one = [
        [c1[i] - c0[i] for i in range(3)] == [1, 1, 1],
        [c2[i] - c1[i] for i in range(3)] == [1, 1, 1]
    ]
    possibility_two = [
        [c1[i] - c0[i] for i in range(3)] == [2, 2, 2],
        [c2[i] - c1[i] for i in range(3)] == [2, 2, 2]
    ]
    if all(possibility_one) or all(possibility_two):
        TileSet.update_used_excluded_stats(sorted_chows)
        return 1
    return 0


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
    if sorted(numbers_list).count(all_ts[0].numbers[0]) >= 3:
        start_ind = 0
    elif sorted(numbers_list).count(all_ts[1].numbers[0]) >= 3:
        start_ind = 1
    else:
        return 0
    if not ((all_ts[start_ind+1].suit != all_ts[start_ind].suit) and
            (all_ts[start_ind+2].suit != all_ts[start_ind+1].suit)):
        return 0
    if not ((all_ts[start_ind+1].numbers[0] == all_ts[start_ind].numbers[0]) and
            (all_ts[start_ind+2].numbers[0] == all_ts[start_ind+1].numbers[0])):
        return 0
    TileSet.update_used_excluded_stats([all_ts[start_ind], all_ts[start_ind+1], all_ts[start_ind+2]])
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
    if (revealed_counter > 1 and usable_sets == 4) or (revealed_counter > 0 and usable_sets == 3):
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
    TileSet.update_used_excluded_stats([chows[start_ind], chows[start_ind+1], chows[start_ind+2]])
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

    suit_map = sorted(list(map(lambda x: x.suit,
                               fresh_pungs + used_but_not_excluded_pungs + fresh_kongs + used_but_not_excluded_kongs)))
    most_suit = suit_map[0] if suit_map.count(suit_map[0]) >= 3 else suit_map[1]

    to_be_used = []
    for ts in fresh_pungs + fresh_kongs:
        if ts.suit == most_suit:
            to_be_used += [ts]

    if len(to_be_used) == 0:
        return 0

    for ts in used_but_not_excluded_pungs + used_but_not_excluded_kongs:
        if ts.suit == most_suit:
            to_be_used += [ts]

    if len(to_be_used) < 3:
        return 0

    sorted_ts = sorted(to_be_used)
    for ts in sorted_ts:
        if ts.suit == "wind" or ts.suit == "dragon":
            return 0
    if not ((int(sorted_ts[1].numbers[0]) == int(sorted_ts[0].numbers[0]) + 1) and
            (int(sorted_ts[2].numbers[0]) == int(sorted_ts[1].numbers[0]) + 1)):
        return 0
    TileSet.update_used_excluded_stats(sorted_ts)
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
        TileSet.update_used_excluded_stats(chows)
        return 1
    return 0


def three_kongs(kongs):
    fresh_kongs = [ts for ts in kongs if (not ts.excluded and not ts.used)]
    used_but_not_excluded_kongs = [ts for ts in kongs if (ts.used and not ts.excluded)]
    if len(fresh_kongs) < 1 or len(kongs) + len(used_but_not_excluded_kongs) < 3:
        return 0
    # If there's 4, 4 kongs will be activated anyways and void this one.
    TileSet.update_used_excluded_stats(fresh_kongs)
    TileSet.update_used_excluded_stats(used_but_not_excluded_kongs)
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
    TileSet.update_used_excluded_stats(chows)
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
    for ts in all_ts:
        if ts.suit == "wind" or ts.suit == "dragon":
            return 0
    if not ((all_ts[1].suit == all_ts[0].suit) and (all_ts[2].suit == all_ts[1].suit)
            and (all_ts[3].suit == all_ts[2].suit)):
        return 0
    if not ((int(all_ts[1].numbers[0]) == int(all_ts[0].numbers[0]) + 1) and
            (int(all_ts[2].numbers[0]) == int(all_ts[1].numbers[0]) + 1) and
            (int(all_ts[3].numbers[0]) == int(all_ts[2].numbers[0]) + 1)):
        return 0
    TileSet.update_used_excluded_stats(kongs)
    TileSet.update_used_excluded_stats(pungs)
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
    TileSet.update_used_excluded_stats([fresh_pungs[i] for i in fp_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_pungs[i] for i in up_indicies_to_update])
    TileSet.update_used_excluded_stats([fresh_kongs[i] for i in fk_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_kongs[i] for i in uk_indicies_to_update])
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
    TileSet.update_used_excluded_stats([fresh_pungs[i] for i in fp_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_pungs[i] for i in up_indicies_to_update])
    TileSet.update_used_excluded_stats([fresh_kongs[i] for i in fk_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_kongs[i] for i in uk_indicies_to_update])
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
        TileSet.update_used_excluded_stats(chows)
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
    TileSet.update_used_excluded_stats([fresh_pungs[i] for i in fp_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_pungs[i] for i in up_indicies_to_update])
    TileSet.update_used_excluded_stats([fresh_kongs[i] for i in fk_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_kongs[i] for i in uk_indicies_to_update])
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
    TileSet.update_used_excluded_stats([fresh_pungs[i] for i in fp_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_pungs[i] for i in up_indicies_to_update])
    TileSet.update_used_excluded_stats([fresh_kongs[i] for i in fk_indicies_to_update])
    TileSet.update_used_excluded_stats([used_but_not_excluded_kongs[i] for i in uk_indicies_to_update])
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
    TileSet.update_used_excluded_stats(kongs)
    return 1
