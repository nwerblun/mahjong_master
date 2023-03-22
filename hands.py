import numpy as np
from utilities import *


class MahjongHands:
    _hand_titles = np.array([
        "Hand Names",
        "2x identical chows, same suit",
        "2x identical chows, 2 suits",
        "6-tile straight, same suit",
        "123 and 789 same suit",
        "Terminal/non-dragon honor pung",
        "Melded (revealed) kong",
        "Voided suit",
        "No honor tiles",
        "Self-drawn",
        "Flowers",
        "Edge wait",
        "Closed wait",
        "Single wait",
        "Dragon pung",
        "Round wind pung",
        "Seat wind pung",
        "Concealed hand, discard win",
        "All chows, no honors",
        "Tile hog",
        "2x pung, same number, diff. suits",
        "2x concealed pung",
        "Concealed kong",
        "All simples (2-8 only)",
        "Outside hand (term/honor in each set)",
        "Fully concealed, self draw win",
        "2x melded kong",
        "Last tile",
        "All pungs",
        "Half flush (one suit + honors)",
        "Mixed shifted chow",
        "All types",
        "Melded hand",
        "2x dragon pung",
        "Mixed straight",
        "Reversible tiles",
        "Mixed triple chow (same chow, 3 suits)",
        "Mixed shifted pungs",
        "2x concealed kong",
        "Last tile draw",
        "Last tile claim",
        "Replacement win",
        "Kong rob",
        "Chicken hand",
        "Lesser Honors + Knitted sequence",
        "Knitted Straight",
        "Upper Four (6-9 suit tiles only)",
        "Lower Four (1-4 suit tiles only)",
        "Big Three Winds (3 wind pung/kongs)",
        "Pure Straight (1-9 same suit)",
        "Three-suited terminal chows",
        "Pure Shifted Chows",
        "All 5s",
        "Triple Pung",
        "3x Concealed Pungs",
        "Seven Pairs",
        "Greater Honors and Knitted Tiles",
        "All Even Pungs",
        "Full Flush (All tiles in the same suit)",
        "Pure Triple Chow",
        "Pure Shifted Pungs",
        "Upper Tiles (7-9 tiles, suits only)",
        "Middle Tiles (4-6 tiles, suits only)",
        "Lower Tiles (1-3 tiles, suits only)",
        "Four Shifted Chows",
        "Three Kongs",
        "All terminals + honors",
        "Quad Chow (4x same chow)",
        "Four Pure Shifted Pungs",
        "All terminals",
        "All Honors",
        "Little 4 Winds",
        "Little 3 Dragons",
        "4x Concealed Pungs",
        "Pure Terminal Chows",
        "Big 4 Winds",
        "Big 3 Dragons",
        "All Green",
        "Nine Gates",
        "Four Kongs",
        "Seven Shifted Pairs",
        "Thirteen Orphans"
    ])
    _points = np.array([
        "Point Values",
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        4,
        6,
        6,
        6,
        6,
        6,
        6,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        12,
        12,
        12,
        12,
        12,
        16,
        16,
        16,
        16,
        16,
        16,
        24,
        24,
        24,
        24,
        24,
        24,
        24,
        24,
        24,
        32,
        32,
        32,
        48,
        48,
        64,
        64,
        64,
        64,
        64,
        64,
        88,
        88,
        88,
        88,
        88,
        88,
        88
    ])
    _notes = np.array([
        "Notes",
        "",
        "",
        "",
        "",
        "Not round/seat, no dragons",
        "",
        "Hand contains 2/3 suits and optionally honors. Missing one suit entirely",
        "",
        "",
        "Flowers do not count towards the minimum 8 points needed to win, but provide 1 point each if you do.",
        ("Winning on 3 to finish a 123 or a 7 to finish a 789 ONLY. Must be the only possible tile to complete 4 SETS "
         + "AND PAIR. Doesn't count for hands that win with non-4 set + pair hands"),
        ("Winning on the middle piece of a chow. E.g. 2 in a 1-2-3 chow. Must be the only possible tile to complete 4"
         + " SETS AND PAIR. Doesn't count for hands that win with non-4 set + pair hands"),
        ("Winning on single tile to form a pair. Must be the only possible tile to complete 4 SETS AND PAIR." +
         " Doesn't count for hands that win with non-4 set + pair hands"),
        "",
        "",
        "",
        "Must win by discard",
        "",
        "Uses all 4 of the same suit tile, but not as a kong. E.g. 123, 333",
        "",
        "",
        "",
        "",
        "Pair must also be terminal/honor",
        "must win by self draw",
        "",
        "can be self drawn or stolen, must be clear from discards/melds and your hand",
        "4 pungs/kongs + pair",
        "one suit only + honors",
        "each chow shifted up by 1. E.g. 456, 567, 678",
        "all sets (including pair) is a unique suit / honor",
        "Must win on discard",
        "",
        "suit 1 - 123 suit 2 - 456 suit 3 - 789",
        "hand contains tiles from 1234589 of dots 245689 bamboo white dragon",
        "each chow must be different suit",
        "3 suits, 3 pungs, each shifted by 1 e.g. 222 333 444",
        "any 2 kongs",
        "self draw the final tile and win",
        "take the final tile after it is discarded and win",
        "declare kong, win off drawn tile. Counts as self-drawn",
        "someone adds a tile to their melded pung to make a kong, but you steal it and win",
        "a hand worth 0 points otherwise",
        ("Suit 1: 147 Suit 2: 248 Suit 3: 369 + all honors = 16 tile set Have 14/16 of this 'set' with NO DUPLICATES."
         + " Knitted sequence doesn't need to be a complete straight. CAN combine with knitted straight."),
        ("Suit 1: 147 Suit 2: 248 Suit 3: 369 Have ALL 1-9 of the knitted set + "
         + "5 unique honors CAN combine with lesser honors + knitted"),
        "Suit tiles >= 6",
        "Suit tiles <= 4",
        "3 wind pung/kong. Doesn't combine with terminal/honor pung unless you have an additional non-wind pung.",
        "",
        "123,789 in suit 1, 123,789 in suit 2 and 55 pair in third suit",
        "Same suit, 3 chows who's sequences are shifted by 1. Example, 123,234,345",
        "All sets (including pair) has a 5",
        "3 pungs, 3 suits, same number in each",
        "",
        "7 different pairs. No actual sets.",
        ("Suit 1: 147 Suit 2: 248 Suit 3: 369 + all honors = 16 tile set Have ALL honors + 7/16 of this 'set' "
         + "with NO DUPLICATES. Knitted sequence doesn't need to be a complete straight. "
         + "CAN combine with knitted straight."),
        "No honor tiles allowed",
        "No honor tiles allowed",
        "Same sequence 3 times, same suit. Looks like a 3x pung, but actually is 3x sequences",
        ("Very similar to triple chow. 3 sequential pungs in the same suit (111 222 333). "
         + "Can also be used for (but not combined with) pure triple chow. This can also use kongs though, "
         + "while triple chow cannot."),
        "Suit tiles >= 7",
        "Suit tiles in 4, 5, 6",
        "Suit tiles <= 3",
        ("4 chows, same suit, each up by 1 OR 2 numbers but not a combination of both. "
         + "Examples: 123 234 345 456 123 345 567 789"),
        "Combines with 3x concealed pung if concealed",
        "",
        "",
        "4x pungs (or kongs) shifted up by 1 each. 111 222 333 444",
        "",
        "",
        "3 wind pungs + wind pair. Can combine with seat/round wind points",
        "2 dragon pungs + pair of the 3rd dragon",
        "",
        "123 123, 789, 789 + 55 pair same suit",
        "pung/kong of all 4 wind tiles",
        "pung/kong of all dragons",
        "23468 of bamboo and/or green dragons only",
        "1112345678999 concealed and winning on a 1-9 of the same suit",
        "",
        "7 pairs each shifted up by 1. Example 22 33 44 55 66 77 88",
        "Singles of all 1,9 from each suit + all honor tiles + 1 duplicate of any"
    ])
    _categories = np.array([
        "Categories",
        "chow, suit",
        "chow, suit",
        "chow, suit, straight",
        "chow, terminal, suit",
        "pung, terminal, honor",
        "kong, melded",
        "suit",
        "honor, dragon, suit",
        "winning",
        "special",  # Flowers
        "winning",
        "winning",
        "winning",
        "pung, honor, dragon",  # Begin 2 point rules
        "pung, honor",
        "pung, honor",
        "winning, concealed",
        "chow, winning, honor",
        "chow, pung, kong",
        "pung, suit",
        "pung, concealed",
        "kong, concealed",
        "winning, suit, honor",
        "winning, honor, terminal",  # Begin 4 point rules
        "winning, concealed",
        "kong, melded",
        "winning",
        "pung, kong",
        "suit, honor",
        "chow, suit",
        "suit, honor",
        "melded, winning",
        "pung, honor, dragon",
        "chow, suit, straight",  # Begin 8 point rules
        "suit",
        "chow, suit, straight",
        "pung, suit",  # Mixed shifted pungs
        "kong, concealed",
        "winning",
        "winning",
        "winning",
        "kong, winning",
        "special, winning",
        "suit, chow, honor, dragon, special",  # Begin 12 point rules
        "suit, chow, honor, dragon, special, straight",
        "suit, pung, chow, kong, terminal",
        "suit, pung, chow, kong, terminal",
        "honor, pung, kong",
        "chow, straight, suit",  # Begin 16 point rules
        "chow, straight, terminal, suit",
        "chow, straight, suit",
        "suit, chow, pung, kong",
        "pung, suit",
        "pung, concealed",
        "special",  # Begin 24 point rules
        "suit, chow, honor, dragon, special",
        "suit, pung",
        "suit, pung, chow, kong",
        "suit, chow, straight",
        "suit, pung, kong",
        "suit, chow, pung, kong, terminal",
        "suit, chow, pung, kong",
        "suit, chow, pung, kong, terminal",
        "chow, suit, straight",  # Begin 32 point rules
        "kong, melded, concealed",
        "honor, pung, kong, chow, terminal",
        "chow, suit",  # Begin 48 point rules
        "pung, suit, kong",
        "terminal, pung, kong",  # Begin 64 point rules
        "honor, pung, kong",
        "honor, pung, kong",
        "honor, dragon, pung, kong",
        "pung, kong, dragon, honor, concealed",
        "chow, suit, terminal",
        "honor, pung, kong",  # Begin 88 point rules
        "honor, dragon, pung, kong",
        "chow, pung, kong, suit",
        "suit, pung, kong, concealed",
        "kong, melded, concealed",
        "suit, special",
        "special, honor, dragon, terminal"
    ])
    _example_images = np.array([
        "[IMAGE]Example Images",
        "./img/examples/pure_double_chow.png",
        "./img/examples/mixed_double_chow.png",
        "./img/examples/short_straight.png",
        "./img/examples/two_terminal_chow.png",
        "./img/examples/terminal_honor_pung.png",
        "./img/examples/melded_kong.png",
        "./img/examples/voided_suit.png",
        "./img/examples/no_honor.png",
        "./img/examples/empty.gif",
        "./img/base_tiles/f1.png",
        "./img/examples/edge_wait.png",
        "./img/examples/closed_wait.png",
        "./img/examples/single_wait.png",
        "./img/examples/dragon_pung.png",
        "./img/examples/round_wind_pung.png",
        "./img/examples/seat_wind_pung.png",
        "./img/examples/empty.gif",
        "./img/examples/all_chows.png",
        "./img/examples/tile_hog.png",
        "./img/examples/double_pung.png",
        "./img/examples/empty.gif",
        "./img/examples/empty.gif",
        "./img/examples/all_simples.png",
        "./img/examples/outside_hand.png",
        "./img/examples/empty.gif",
        "./img/examples/two_kongs.png",
        "./img/examples/empty.gif",
        "./img/examples/all_pungs.png",
        "./img/examples/half_flush.png",
        "./img/examples/mixed_shifted_chows.png",
        "./img/examples/all_types.png",
        "./img/examples/empty.gif",
        "./img/examples/two_dragon_pungs.png",
        "./img/examples/mixed_straight.png",
        "./img/examples/reversible_tiles.png",
        "./img/examples/mixed_triple_chow.png",
        "./img/examples/mixed_shifted_pungs.png",
        "./img/examples/empty.gif",
        "./img/examples/empty.gif",
        "./img/examples/empty.gif",
        "./img/examples/empty.gif",
        "./img/examples/empty.gif",
        "./img/examples/chicken_hand.png",
        "./img/examples/lesser_honors_and_knitted_tiles.png",
        "./img/examples/knitted_straight.png",
        "./img/examples/upper_four.png",
        "./img/examples/lower_four.png",
        "./img/examples/big_three_winds.png",
        "./img/examples/pure_straight.png",
        "./img/examples/three_suited_terminal_chows.png",
        "./img/examples/pure_shifted_chows.png",
        "./img/examples/all_fives.png",
        "./img/examples/triple_pung.png",
        "./img/examples/empty.gif",
        "./img/examples/seven_pairs.png",
        "./img/examples/greater_honors_and_knitted_tiles.png",
        "./img/examples/all_even_pungs.png",
        "./img/examples/full_flush.png",
        "./img/examples/pure_triple_chow.png",
        "./img/examples/pure_shifted_pungs.png",
        "./img/examples/upper_tiles.png",
        "./img/examples/middle_tiles.png",
        "./img/examples/lower_tiles.png",
        "./img/examples/four_shifted_chows.png",
        "./img/examples/empty.gif",
        "./img/examples/all_terminals_and_honors.png",
        "./img/examples/quadruple_chow.png",
        "./img/examples/four_pure_shifted_pungs.png",
        "./img/examples/all_terminals.png",
        "./img/examples/all_honors.png",
        "./img/examples/little_four_winds.png",
        "./img/examples/little_three_dragons.png",
        "./img/examples/empty.gif",
        "./img/examples/pure_terminal_chows.png",
        "./img/examples/big_four_winds.png",
        "./img/examples/big_three_dragons.png",
        "./img/examples/all_green.png",
        "./img/examples/nine_gates.png",
        "./img/examples/empty.gif",
        "./img/examples/seven_shifted_pairs.png",
        "./img/examples/thirteen_orphans.png"
    ])
    # https://playmahjong.io/chinese-official-rules for voiding
    _voided_by = np.array([
        "Voided By",
        ", ".join([_hand_titles[59], _hand_titles[67], _hand_titles[74]]),
        ", ".join([_hand_titles[36], _hand_titles[50]]),
        ", ".join([_hand_titles[49], _hand_titles[64]]),
        ", ".join([_hand_titles[49], _hand_titles[50], _hand_titles[74]]),
        ", ".join([_hand_titles[66], _hand_titles[69], _hand_titles[70], _hand_titles[71], _hand_titles[75], _hand_titles[78]]),
        ", ".join([_hand_titles[26], _hand_titles[65], _hand_titles[79]]),
        ", ".join([_hand_titles[29], _hand_titles[35], _hand_titles[58], _hand_titles[70], _hand_titles[74], _hand_titles[78], _hand_titles[80]]),
        ", ".join([_hand_titles[18], _hand_titles[23], _hand_titles[46], _hand_titles[47], _hand_titles[50], _hand_titles[52], _hand_titles[57], _hand_titles[58], _hand_titles[61], _hand_titles[62], _hand_titles[63], _hand_titles[69], _hand_titles[80]]),  # Index 8
        ", ".join([_hand_titles[39], _hand_titles[41]]),
        "",  # Index 10
        "",
        "",
        ", ".join([_hand_titles[32], _hand_titles[55], _hand_titles[79], _hand_titles[80]]),
        ", ".join([_hand_titles[33], _hand_titles[72], _hand_titles[76]]),
        ", ".join([_hand_titles[75]]),
        ", ".join([_hand_titles[75]]),
        ", ".join([_hand_titles[44], _hand_titles[55], _hand_titles[56], _hand_titles[73], _hand_titles[78], _hand_titles[80], _hand_titles[81]]),
        ", ".join([_hand_titles[50], _hand_titles[74]]),
        ", ".join([_hand_titles[67]]),
        ", ".join([_hand_titles[53]]),  # Index 20
        ", ".join([_hand_titles[21], _hand_titles[54], _hand_titles[73]]),
        ", ".join([_hand_titles[21]]),
        ", ".join([_hand_titles[52], _hand_titles[57], _hand_titles[62]]),
        ", ".join([_hand_titles[66], _hand_titles[69], _hand_titles[70], _hand_titles[81]]),
        "",
        ", ".join([_hand_titles[65], _hand_titles[79]]),
        ", ".join([_hand_titles[42]]),
        ", ".join([_hand_titles[57], _hand_titles[66], _hand_titles[68], _hand_titles[69], _hand_titles[70], _hand_titles[73], _hand_titles[75], _hand_titles[79]]),
        ", ".join([_hand_titles[58], _hand_titles[74], _hand_titles[78], _hand_titles[80]]),
        "",  # Index 30
        ", ".join([_hand_titles[44], _hand_titles[56], _hand_titles[81]]),
        "",
        ", ".join([_hand_titles[72], _hand_titles[76]]),
        "",
        "",
        "",
        "",
        "",
        "",
        "",  # Index 40
        "",
        "",
        "",
        ", ".join([_hand_titles[56]]),
        ", ".join([_hand_titles[61]]),
        ", ".join([_hand_titles[63]]),
        "",
        ", ".join([_hand_titles[71], _hand_titles[75]]),
        "",
        "",  # Index 50
        ", ".join([_hand_titles[64]]),
        "",
        "",
        ", ".join([_hand_titles[73]]),
        ", ".join([_hand_titles[80]]),
        "",
        "",
        ", ".join([_hand_titles[74], _hand_titles[78], _hand_titles[80]]),
        ", ".join([_hand_titles[67]]),
        ", ".join([_hand_titles[68]]),  # Index 60
        "",
        "",
        "",
        "",
        ", ".join([_hand_titles[79]]),
        ", ".join([_hand_titles[69], _hand_titles[70], _hand_titles[81]]),
        "",
        "",
        "",
        "",  # Index 70
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    ])
    tile_names = [
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
    tile_pic_files = ["./img/base_tiles/"+n+".png" for n in tile_names]
    special_tile_names = [
        "f1",
        "f2",
        "f3",
        "f4",
        "z1"
    ]
    special_tile_pic_files = ["./img/base_tiles/"+n+".png" for n in special_tile_names]

    _categories = sort_list(_categories)
    hands_info = np.vstack((_hand_titles, _points, _categories, _notes, _example_images, _voided_by)).T

    @staticmethod
    def get_hand_titles():
        return MahjongHands._hand_titles[1:]

    @staticmethod
    def get_point_values():
        return MahjongHands._points[1:]

    @staticmethod
    def get_voids():
        return MahjongHands._voided_by[1:]

    @staticmethod
    def get_void_tile_photo():
        return MahjongHands.special_tile_pic_files[-1]
