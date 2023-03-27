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
        self.round_wind = "East"
        self.seat_wind = "East"
        self.tileset_format_round_wind = "e"
        self.tileset_format_seat_wind = "e"

    def set_hand(self, concealed_tile_names, revealed_tile_names, drawn_tile,
                 declared_concealed_kongs, revealed_kongs, round_wind, seat_wind):
        # Handle adding/removing tiles typed in by the user
        self.round_wind = round_wind
        self.seat_wind = seat_wind
        self.tileset_format_round_wind = round_wind[0].lower()
        self.tileset_format_seat_wind = seat_wind[0].lower()
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
        # check special cases

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

            # 81. Thirteen Orphans, computed separately
            # 80. Seven Shifted Pairs, computed separately
            hand_dict["point_conditions"][78] = four_kongs(tilesets["kongs"])
            print("Four Kongs: ", str(hand_dict["point_conditions"][78]))

            hand_dict["point_conditions"][77] = nine_gates(tilesets["pungs"], tilesets["kongs"],
                                                           tilesets["chows"], hand_dict["pair"],
                                                           self.pwh.num_suits_used, self.pwh.get_num_honor_tiles())
            print("Nine Gates: ", str(hand_dict["point_conditions"][77]))

            hand_dict["point_conditions"][76] = all_green(tilesets["pungs"], tilesets["kongs"],
                                                          tilesets["chows"], hand_dict["pair"])
            print("All Green: ", str(hand_dict["point_conditions"][76]))

            hand_dict["point_conditions"][75] = big_three_dragons(tilesets["pungs"], tilesets["kongs"])
            print("Big Three Dragons: ", str(hand_dict["point_conditions"][75]))

            hand_dict["point_conditions"][74] = big_four_winds(tilesets["pungs"], tilesets["kongs"])
            print("Big Four Winds: ", str(hand_dict["point_conditions"][74]))

            hand_dict["point_conditions"][73] = pure_terminal_chows(tilesets["chows"],
                                                                    self.pwh.num_suits_used,
                                                                    self.pwh.get_num_honor_tiles(), hand_dict["pair"])
            print("Pure Terminal Chows: ", str(hand_dict["point_conditions"][73]))

            hand_dict["point_conditions"][72] = four_concealed_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Four Concealed Pungs", str(hand_dict["point_conditions"][72]))

            hand_dict["point_conditions"][71] = little_three_dragons(tilesets["pungs"], tilesets["kongs"],
                                                                     hand_dict["pair"])
            print("Little Three Dragons: ", str(hand_dict["point_conditions"][71]))

            hand_dict["point_conditions"][70] = little_four_winds(tilesets["pungs"],
                                                                  tilesets["kongs"], hand_dict["pair"])
            print("Little Four Winds: ", str(hand_dict["point_conditions"][70]))

            # ALL HONORS
            hand_dict["point_conditions"][69] = 1 if (self.pwh.num_suits_used == 0) else 0
            print("All Honors: ", str(hand_dict["point_conditions"][69]))

            hand_dict["point_conditions"][68] = all_terminals(tilesets["chows"], tilesets["pungs"],
                                                              tilesets["kongs"], hand_dict["pair"])
            print("All terminals: ", str(hand_dict["point_conditions"][68]))

            hand_dict["point_conditions"][67] = four_pure_shifted_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Four Pure Shifted Pungs: ", str(hand_dict["point_conditions"][67]))

            hand_dict["point_conditions"][66] = quad_chow(tilesets["chows"])
            print("Quad Chow: ", str(hand_dict["point_conditions"][66]))

            hand_dict["point_conditions"][65] = all_terminals_and_honors(tilesets["chows"], tilesets["pungs"],
                                                                         tilesets["kongs"], hand_dict["pair"])
            print("All Terminals + Honors: ", str(hand_dict["point_conditions"][65]))

            hand_dict["point_conditions"][64] = three_kongs(tilesets["kongs"])
            print("Three Kongs: ", str(hand_dict["point_conditions"][64]))

            hand_dict["point_conditions"][63] = four_shifted_chows(tilesets["chows"])
            print("Four Shifted Chows: ", str(hand_dict["point_conditions"][63]))

            hand_dict["point_conditions"][62] = lower_tiles(tilesets["chows"], tilesets["pungs"],
                                                            tilesets["kongs"], hand_dict["pair"])
            print("Lower Tiles: ", str(hand_dict["point_conditions"][62]))

            hand_dict["point_conditions"][61] = middle_tiles(tilesets["chows"], tilesets["pungs"],
                                                             tilesets["kongs"], hand_dict["pair"])
            print("Middle Tiles: ", str(hand_dict["point_conditions"][61]))

            hand_dict["point_conditions"][60] = upper_tiles(tilesets["chows"], tilesets["pungs"],
                                                            tilesets["kongs"], hand_dict["pair"])
            print("Upper Tiles: ", str(hand_dict["point_conditions"][60]))

            hand_dict["point_conditions"][59] = pure_shifted_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Pure Shifted Pungs: ", str(hand_dict["point_conditions"][59]))

            hand_dict["point_conditions"][58] = pure_triple_chow(tilesets["chows"])
            print("Pure Triple Chow: ", str(hand_dict["point_conditions"][58]))

            # 58. FULL FLUSH
            hand_dict["point_conditions"][57] = 1 if (self.pwh.num_suits_used == 1 and
                                                      self.pwh.get_num_honor_tiles() == 0) else 0
            print("Full Flush: ", str(hand_dict["point_conditions"][57]))

            hand_dict["point_conditions"][56] = all_even_pungs(tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])
            print("All Even Pungs: ", str(hand_dict["point_conditions"][56]))

            # 56. Greater honors + knitted, computed separately
            # 55. Seven pairs, computed separately

            hand_dict["point_conditions"][53] = three_concealed_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Three Concealed Pungs: ", str(hand_dict["point_conditions"][53]))

            hand_dict["point_conditions"][52] = triple_pung(tilesets["pungs"], tilesets["kongs"])
            print("Triple Pungs: ", str(hand_dict["point_conditions"][52]))

            hand_dict["point_conditions"][51] = all_fives(tilesets["chows"], tilesets["pungs"],
                                                          tilesets["kongs"], hand_dict["pair"])
            print("All Fives: ", str(hand_dict["point_conditions"][51]))

            hand_dict["point_conditions"][50] = pure_shifted_chows(tilesets["chows"])
            print("Pure Shifted Chows: ", str(hand_dict["point_conditions"][50]))

            hand_dict["point_conditions"][49] = three_suited_terminal_chows(tilesets["chows"], hand_dict["pair"])
            print("3 Suited Terminal Chows: ", str(hand_dict["point_conditions"][49]))

            hand_dict["point_conditions"][48] = pure_straight(tilesets["chows"])
            print("Pure Straight: ", str(hand_dict["point_conditions"][48]))

            hand_dict["point_conditions"][47] = big_three_winds(tilesets["pungs"], tilesets["kongs"])
            print("Big Three Winds: ", str(hand_dict["point_conditions"][47]))

            hand_dict["point_conditions"][46] = lower_four(tilesets["pungs"], tilesets["kongs"],
                                                           tilesets["chows"], hand_dict["pair"])
            print("Lower Four: ", str(hand_dict["point_conditions"][46]))

            hand_dict["point_conditions"][45] = upper_four(tilesets["pungs"], tilesets["kongs"],
                                                           tilesets["chows"], hand_dict["pair"])
            print("Upper Four: ", str(hand_dict["point_conditions"][45]))

            hand_dict["point_conditions"][44] = 1 if hand_dict["knitted_straight"] else 0
            print("Knitted Straight: ", str(hand_dict["point_conditions"][44]))

            # 44. Lesser Honors + Knitted Tiles, computed separately

            # 43. CHICKEN HAND WOULD GO HERE BUT YOU NEED TO CHECK SPECIAL CASES
            # 42. SPACE FOR KONG ROB
            # 41. SPACE FOR REPLACEMENT WIN
            # 40. SPACE FOR LAST TILE CLAIM
            # 39. SPACE FOR LAST TILE DRAW

            hand_dict["point_conditions"][37] = two_concealed_kongs(tilesets["kongs"])
            print("Two Concealed Kongs: ", str(hand_dict["point_conditions"][37]))

            hand_dict["point_conditions"][36] = mixed_shifted_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Mixed Shifted Pungs: ", str(hand_dict["point_conditions"][36]))

            hand_dict["point_conditions"][35] = mixed_triple_chow(tilesets["chows"])
            print("Mixed Triple Chow: ", str(hand_dict["point_conditions"][35]))

            hand_dict["point_conditions"][34] = reversible_tiles(tilesets["chows"], tilesets["pungs"],
                                                                 tilesets["kongs"], hand_dict["pair"])
            print("Reversible Tiles: ", str(hand_dict["point_conditions"][34]))

            hand_dict["point_conditions"][33] = mixed_straight(tilesets["chows"])
            print("Mixed Straight: ", str(hand_dict["point_conditions"][33]))

            hand_dict["point_conditions"][32] = two_dragon_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Two Dragon Pungs: ", str(hand_dict["point_conditions"][32]))

            # 32. MELDED HAND
            # amt = 1 if (self.pwh.drawn_tile is None and self.pwh.get_num_revealed_sets() == 4) else 0
            # print("Melded Hand:", str(amt))

            # 31. ALL TYPES
            hand_dict["point_conditions"][30] = 1 if (self.pwh.get_num_dragons() > 0 and self.pwh.get_num_winds() > 0
                                                      and self.pwh.num_suits_used == 3) else 0
            print("All Types: ", str(hand_dict["point_conditions"][30]))

            hand_dict["point_conditions"][29] = mixed_shifted_chow(tilesets["chows"])
            print("Mixed shifted chow: ", str(hand_dict["point_conditions"][29]))

            # 29. HALF FLUSH
            hand_dict["point_conditions"][28] = 1 if (self.pwh.num_suits_used == 1 and
                                                      self.pwh.get_num_honor_tiles() > 0) else 0
            print("Half Flush: ", str(hand_dict["point_conditions"][28]))

            # 28. ALL PUNGS
            hand_dict["point_conditions"][27] = 1 if (len(tilesets["chows"]) == 0) else 0
            print("All Pungs (or kongs):", str(hand_dict["point_conditions"][27]))

            # 27. SPACE FOR LAST TILE

            hand_dict["point_conditions"][25] = two_melded_kongs(tilesets["kongs"])
            print("Two Melded Kongs (or 1 melded + 1 concealed): ", str(hand_dict["point_conditions"][25]))

            # 25. FULLY CONCEALED SELF DRAWN
            # amt = 1 if self.pwh.is_fully_concealed() else 0
            # print("Fully concealed, self-drawn win:", str(amt))

            hand_dict["point_conditions"][23] = outside_hand(tilesets["chows"], tilesets["pungs"],
                                                             tilesets["kongs"], hand_dict["pair"])
            print("Outside Hand: ", str(hand_dict["point_conditions"][23]))

            hand_dict["point_conditions"][22] = all_simples(tilesets["chows"], tilesets["pungs"],
                                                            tilesets["kongs"], hand_dict["pair"])
            print("All Simples: ", str(hand_dict["point_conditions"][22]))

            hand_dict["point_conditions"][21] = one_concealed_kong(tilesets["kongs"])
            print("One Concealed Kong: ", str(hand_dict["point_conditions"][21]))

            hand_dict["point_conditions"][20] = two_concealed_pungs(tilesets["pungs"], tilesets["kongs"])
            print("Two Concealed Pungs: ", str(hand_dict["point_conditions"][20]))

            hand_dict["point_conditions"][19] = mixed_double_pung(tilesets["pungs"], tilesets["kongs"])
            print("Mixed Double Pungs: ", str(hand_dict["point_conditions"][19]))

            hand_dict["point_conditions"][18] = tile_hog(tilesets["pungs"], tilesets["kongs"],
                                                         hand_dict["pair"], tilesets["chows"])
            print("Tile Hog: ", str(hand_dict["point_conditions"][18]))

            hand_dict["point_conditions"][17] = all_chow_no_honors(tilesets["pungs"], tilesets["kongs"],
                                                                   hand_dict["pair"])
            print("All Chows (no honors): ", str(hand_dict["point_conditions"][17]))

            # 17. CONCEALED HAND DISCARD WIN
            # amt = 1 if (self.pwh.is_fully_concealed() and self.pwh.drawn_tile is None) else 0
            # print("Fully concealed, stolen discard win:", str(amt))

            hand_dict["point_conditions"][15] = seat_wind_pung(tilesets["pungs"], tilesets["kongs"],
                                                               self.tileset_format_seat_wind)
            print("Seat Wind Pung: ", str(hand_dict["point_conditions"][15]))

            hand_dict["point_conditions"][14] = round_wind_pung(tilesets["pungs"], tilesets["kongs"], self.tileset_format_round_wind)
            print("Round Wind Pung: ", str(hand_dict["point_conditions"][14]))

            hand_dict["point_conditions"][13] = dragon_pung(tilesets["pungs"], tilesets["kongs"])
            print("Dragon Pung: ", str(hand_dict["point_conditions"][13]))

            # 13. SPACE FOR SINGLE WAIT
            # 12. SPACE FOR CLOSED WAIT
            # 11. SPACE FOR EDGE WAIT
            # 10. SPACE FOR FLOWERS???

            # 09. SELF DRAWN
            # amt = 1 if self.pwh.drawn_tile is not None else 0
            # print("Self drawn:", str(amt))

            # 08. NO HONOR TILES
            hand_dict["point_conditions"][7] = 1 if (self.pwh.get_num_honor_tiles() == 0) else 0
            print("No Honor Tiles:", str(hand_dict["point_conditions"][7]))

            # 07. VOIDED SUIT
            hand_dict["point_conditions"][6] = 1 if (self.pwh.num_suits_used <= 2) else 0
            print("Voided Suit:", str(hand_dict["point_conditions"][6]))

            hand_dict["point_conditions"][5] = melded_kong(tilesets["kongs"])
            print("Melded Kong: ", str(hand_dict["point_conditions"][5]))

            hand_dict["point_conditions"][4] = terminal_non_dragon_honor_pung(tilesets["pungs"], tilesets["kongs"],
                                                                              self.tileset_format_seat_wind,
                                                                              self.tileset_format_round_wind)
            print("Terminal/Honor Pung (no dragon/seat/round wind): ", str(hand_dict["point_conditions"][4]))

            hand_dict["point_conditions"][3] = two_terminal_chows(tilesets["chows"])
            print("Two Terminal Chows: ", str(hand_dict["point_conditions"][3]))

            hand_dict["point_conditions"][2] = short_straight(tilesets["chows"])
            print("Short Straight: ", str(hand_dict["point_conditions"][2]))

            hand_dict["point_conditions"][1] = mixed_double_chow(tilesets["chows"])
            print("Mixed double chow: ", str(hand_dict["point_conditions"][1]))

            hand_dict["point_conditions"][0] = pure_double_chow(tilesets["chows"])
            print("Pure double chow: ", str(hand_dict["point_conditions"][0]))

        lhks = lesser_honors_knitted_seq(self.pwh)
        sp = seven_pairs(self.pwh)
        ghkt = greater_honors_knitted_tiles(self.pwh)
        ssp = seven_shifted_pairs(self.pwh)
        to = thirteen_orphans(self.pwh)
        print("Thirteen orphans: ", str(to))
        print("Seven Shifted Pairs: ", str(ssp))
        print("Greater Honors + Knitted: ", str(ghkt))
        print("Seven Pairs: ", str(sp))
        print("Lesser Honors + Knitted: ", str(lhks))
        # after checking all hands
        # check chicken hand




