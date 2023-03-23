from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from game import *
from pathfinding import *
from hands import MahjongHands
from scoring_conditions import *


class Calculator:
    def __init__(self):
        self.hand = Hand()
        self.pwh = None
        self.hand_titles = MahjongHands.get_hand_titles()
        self.official_point_values = MahjongHands.get_point_values()
        self.voids = MahjongHands.get_voids()

    def set_hand(self, concealed_tile_names, revealed_tile_names, drawn_tile, declared_concealed_kongs, revealed_kongs):
        # Handle adding/removing tiles typed in by the user
        self.hand.clear_hand()
        for n in concealed_tile_names:
            self.hand.add_tile_to_hand(False, n, more=True)
        for n in revealed_tile_names:
            self.hand.add_tile_to_hand(True, n, more=True)
        for k in declared_concealed_kongs:
            self.hand.add_declared_concealed_kong_to_hand(k)
        for k in revealed_kongs:
            self.hand.add_revealed_kong_to_hand(k)
        self.hand.set_drawn_tile(drawn_tile)
        self.pwh = PossibleWinningHand(self.hand)
        self._score_winning_sets()

    def _check_special_cases(self):
        return self

    def _score_winning_sets(self):
        if self.pwh.get_num_tiles_in_hand() < 14:
            return
        # Format (used=False, excluded=False)
        # if used to make a set, used=True.
        # If used with a set that has used=True, excluded = True.
        # If attempting to use with a set that is excluded, don't.
        # Go through conditions in reverse order so exclusionary rule skips cheap fans.
        print("======================================")
        for hand_dict in self.pwh.four_set_pair_hands:
            tilesets = {"kongs": [], "chows": [], "pungs": []}
            for s in hand_dict["declared_concealed_kongs"]:
                tilesets["kongs"] += [TileSet(s, "kong", concealed=True, declared=True)]
            for s in hand_dict["concealed_kongs"]:
                tilesets["kongs"] += [TileSet(s, "kong", concealed=True, declared=False)]
            for s in hand_dict["revealed_kongs"]:
                tilesets["kongs"] += [TileSet(s, "kong")]
            for s in hand_dict["concealed_chows"]:
                tilesets["chows"] += [TileSet(s, "chow", concealed=True)]
            for s in hand_dict["revealed_chows"]:
                tilesets["chows"] += [TileSet(s, "chow")]
            for s in hand_dict["concealed_pungs"]:
                tilesets["pungs"] += [TileSet(s, "pung", concealed=True)]
            for s in hand_dict["revealed_pungs"]:
                tilesets["pungs"] += [TileSet(s, "pung")]

            # TODO: Figure out which index
            # hand_dict["point_conditions"]
            print("Knitted Straight: ", str(hand_dict["knitted_straight"]))

            amt = four_kongs(tilesets["kongs"])
            print("Four Kongs: ", str(amt))

            amt = all_green(tilesets["pungs"], tilesets["kongs"], tilesets["chows"], hand_dict["pair"])
            print("All Green: ", str(amt))

            amt = nine_gates(tilesets["pungs"], tilesets["kongs"], tilesets["chows"], hand_dict["pair"],
                             self.pwh.num_suits_used, self.pwh.get_num_honor_tiles())
            print("Nine Gates: ", str(amt))

            amt = big_three_dragons(tilesets["pungs"], tilesets["kongs"])
            print("Big Three Dragons: ", str(amt))

            amt = big_four_winds(tilesets["pungs"], tilesets["kongs"])
            print("Big Four Winds: ", str(amt))

            amt = pure_terminal_chows(tilesets["chows"],
                                      self.pwh.num_suits_used, self.pwh.get_num_honor_tiles(), hand_dict["pair"])
            print("Pure Terminal Chows: ", str(amt))

            amt = four_concealed_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Four Concealed Pungs", str(amt))

            amt = little_three_dragons(tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("Little Three Dragons: ", str(amt))

            amt = little_four_winds(tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("Little Four Winds: ", str(amt))

            amt = 1 if (self.pwh.num_suits_used == 0) else 0
            print("All Honors: ", str(amt))

            amt = all_terminals(tilesets["chows"], tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("All terminals: ", str(amt))

            amt = four_pure_shifted_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Four Pure Shifted Pungs: ", str(amt))

            amt = quad_chow(tilesets["chows"])
            print("Quad Chow: ", str(amt))

            amt = all_terminals_and_honors(tilesets["chows"], tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("All Terminals + Honors: ", str(amt))

            amt = three_kongs(tilesets["kongs"])
            print("Three Kongs: ", str(amt))

            amt = four_shifted_chows(tilesets["chows"])
            print("Four Shifted Chows: ", str(amt))

            amt = lower_tiles(tilesets["chows"], tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("Lower Tiles: ", str(amt))

            amt = middle_tiles(tilesets["chows"], tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("Middle Tiles: ", str(amt))

            amt = upper_tiles(tilesets["chows"], tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("Upper Tiles: ", str(amt))

            amt = pure_shifted_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Pure Shifted Pungs: ", str(amt))

            amt = pure_triple_chow(tilesets["chows"])
            print("Pure Triple Chow: ", str(amt))

            amt = 1 if (self.pwh.num_suits_used == 1 and self.pwh.get_num_honor_tiles() == 0) else 0
            print("Full Flush: ", str(amt))

            amt = all_even_pungs(tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("All Even Pungs: ", str(amt))

            amt = three_concealed_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Three Concealed Pungs: ", str(amt))

            amt = triple_pung(tilesets["pungs"], tilesets["kongs"])
            print("Triple Pungs: ", str(amt))

            amt = all_fives(tilesets["chows"], tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("All Fives: ", str(amt))

            amt = pure_shifted_chows(tilesets["chows"])
            print("Pure Shifted Chows: ", str(amt))

            amt = three_suited_terminal_chows(tilesets["chows"], hand_dict["pair"])
            print("3 Suited Terminal Chows: ", str(amt))

            amt = pure_straight(tilesets["chows"])
            print("Pure Straight: ", str(amt))

            amt = big_three_winds(tilesets["pungs"], tilesets["kongs"])
            print("Big Three Winds: ", str(amt))

            amt = lower_four(tilesets["pungs"], tilesets["kongs"], tilesets["chows"], hand_dict["pair"])
            print("Lower Four: ", str(amt))

            amt = upper_four(tilesets["pungs"], tilesets["kongs"], tilesets["chows"], hand_dict["pair"])
            print("Upper Four: ", str(amt))

            # CHICKEN HAND WOULD GO HERE BUT YOU NEED TO CHECK SPECIAL CASES
            # SPACE FOR KONG ROB
            # SPACE FOR REPLACEMENT WIN
            # SPACE FOR LAST TILE CLAIM
            # SPACE FOR LAST TILE DRAW

            amt = two_concealed_kongs(tilesets["kongs"])
            print("Two Concealed Kongs: ", str(amt))

            amt = mixed_shifted_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Mixed Shifted Pungs: ", str(amt))

            amt = mixed_double_chow(tilesets["chows"])
            print("Mixed double chow: ", str(amt))
            hand_dict["point_conditions"][1] = amt

            amt = pure_double_chow(tilesets["chows"])
            print("Pure double chow: ", str(amt))
            hand_dict["point_conditions"][0] = amt

            print("-----next hand------")

        # check special cases
        amt = lesser_honors_knitted_seq(self.pwh)
        print("Lesser Honors + Knitted: ", str(amt))
        amt = seven_pairs(self.pwh)
        print("Seven Pairs: ", str(amt))
        amt = greater_honors_knitted_tiles(self.pwh)
        print("Greater Honors + Knitted: ", str(amt))
        amt = seven_shifted_pairs(self.pwh)
        print("Seven Shifted Pairs: ", str(amt))
        amt = thirteen_orphans(self.pwh)
        print("Thirteen orphans: ", str(amt))
        # after checking all hands
        # check chicken hand




