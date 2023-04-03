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
        self.knitted_straight = False
        self.round_wind = "East"
        self.seat_wind = "East"
        self.tileset_format_round_wind = "e"
        self.tileset_format_seat_wind = "e"
        self.drew_last_tile = False
        self.last_tile_of_its_kind = False
        self.win_on_replacement = False
        self.robbed_kong = False
        self.score_breakdown = ""
        self.total_hand_value = 0

    def set_special_conditions(self, drew_last_tile, last_tile_of_its_kind, win_on_replacement, robbed_kong):
        self.drew_last_tile = drew_last_tile
        self.last_tile_of_its_kind = last_tile_of_its_kind
        self.win_on_replacement = win_on_replacement
        self.robbed_kong = robbed_kong

    def set_hand(self, concealed_tile_names, revealed_tile_names, final_tile, self_drawn_final,
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
        self.hand.set_final_tile(final_tile, self_drawn_final)
        self.pwh = PossibleWinningHand(self.hand)
        self.total_hand_value, self.score_breakdown = self.get_score_summary()

    def get_score_summary(self):
        if self.pwh.get_num_tiles_in_hand() < 14:
            return 0, ""
        max_point_array = self._score_winning_sets()
        max_special_score = self._score_special_sets()
        max_point_array = self._chicken_hand(max_point_array, max_special_score)
        for i in range(len(max_point_array)):
            if max_point_array[i] > 0:
                if self.voids[i] != "":
                    voided_by = self.voids[i].split(", ")
                    for j in range(len(voided_by)):
                        void_index = get_index_of(self.hand_titles, voided_by[j])
                        if max_point_array[void_index] > 0:
                            max_point_array[i] = 0
        total = 0
        breakdown = ""
        for i in range(len(max_point_array)):
            if max_point_array[i] > 0:
                total += int(self.official_point_values[i]) * int(max_point_array[i])
                str_to_print = self.hand_titles[i]
                str_to_print += ","+"x" + str(max_point_array[i])
                str_to_print += ",=,"
                str_to_print += str(int(self.official_point_values[i])*max_point_array[i])
                breakdown += str_to_print + "\n"
        breakdown += "TOTAL HAND VALUE,,=," + str(total)
        return total, breakdown

    def _score_special_sets(self):
        base_array = [0] * len(MahjongHands.get_hand_titles())
        max_score_array = base_array[:]
        lhks = lesser_honors_knitted_seq(self.pwh)
        if self.pwh._has_knitted_straight(self.pwh.concealed_tiles)[0]:
            if lhks + 12 > sum(max_score_array) and lhks > 0:
                max_score_array[44] = 1
                max_score_array[43] = 1
        else:
            if lhks > sum(max_score_array):
                max_score_array[43] = 1

        sp = seven_pairs(self.pwh)
        if sp > sum(max_score_array):
            max_score_array = base_array[:]
            max_score_array[54] = 1
        if sp > 0:
            seen = []
            for t in self.pwh.concealed_tiles:
                if self.pwh.concealed_tiles.count(t) == 4 and t not in seen:
                    seen += [t]
                    max_score_array[18] += 1

        ghkt = greater_honors_knitted_tiles(self.pwh)
        if ghkt > sum(max_score_array):
            max_score_array = base_array[:]
            max_score_array[55] = 1

        ssp = seven_shifted_pairs(self.pwh)
        if ssp > sum(max_score_array):
            max_score_array = base_array[:]
            max_score_array[79] = 1

        to = thirteen_orphans(self.pwh)
        if to > sum(max_score_array):
            max_score_array = base_array[:]
            max_score_array[80] = 1

        if sum(max_score_array) == 0:
            return base_array

        # ALL HONORS
        max_score_array[69] = 1 if (self.pwh.num_suits_used == 0) else 0
        # 58. FULL FLUSH
        max_score_array[57] = 1 if (self.pwh.num_suits_used == 1 and
                                    self.pwh.get_num_honor_tiles() == 0) else 0

        # 42. KONG ROB
        max_score_array[41] = 1 if self.robbed_kong else 0

        # 41. REPLACEMENT WIN
        max_score_array[40] = 1 if self.win_on_replacement else 0

        # 40. LAST TILE CLAIM
        max_score_array[39] = 1 if self.drew_last_tile and not self.pwh.self_drawn_final_tile else 0

        # 39. LAST TILE DRAW
        max_score_array[38] = 1 if self.drew_last_tile and self.pwh.self_drawn_final_tile else 0

        # 32. MELDED HAND
        if not self.pwh.self_drawn_final_tile and self.pwh.get_num_revealed_sets() == 4 and self.pwh.single_wait:
            max_score_array[31] = 1

        # 31. ALL TYPES
        max_score_array[30] = 1 if (self.pwh.get_num_dragons() > 0 and self.pwh.get_num_winds() > 0
                                    and self.pwh.num_suits_used == 3) else 0

        # 29. HALF FLUSH
        max_score_array[28] = 1 if (self.pwh.num_suits_used == 1 and
                                    self.pwh.get_num_honor_tiles() > 0) else 0

        # 27. LAST TILE
        max_score_array[26] = 1 if self.last_tile_of_its_kind else 0

        # 25. FULLY CONCEALED SELF DRAWN
        if self.pwh.self_drawn_final_tile and self.pwh.is_fully_concealed():
            max_score_array[24] = 1

        # 17. CONCEALED HAND DISCARD WIN
        if self.pwh.is_fully_concealed() and not self.pwh.self_drawn_final_tile:
            max_score_array[16] = 1

        # 13. SINGLE WAIT
        if self.pwh.single_wait:
            max_score_array[12] = 1

        # 12. CLOSED WAIT
        if self.pwh.closed_wait:
            max_score_array[11] = 1

        # 11. EDGE WAIT
        if self.pwh.edge_wait:
            max_score_array[10] = 1

        # 09. SELF DRAWN
        if self.pwh.self_drawn_final_tile:
            max_score_array[8] = 1

        # 08. NO HONOR TILES
        max_score_array[7] = 1 if (self.pwh.get_num_honor_tiles() == 0) else 0

        # 07. VOIDED SUIT
        max_score_array[6] = 1 if (self.pwh.num_suits_used <= 2) else 0
        return max_score_array

    def _chicken_hand(self, max_score_arr, max_special_score_arr):
        base_array = [0] * len(MahjongHands.get_hand_titles())
        if self.pwh.get_num_tiles_in_hand() < 14:
            return base_array
        if sum(max_score_arr) == 0 and sum(max_special_score_arr) == 0:
            base_array[42] = 1
            return base_array
        if sum(max_score_arr) > sum(max_special_score_arr):
            return max_score_arr
        if sum(max_special_score_arr) > sum(max_score_arr):
            return max_special_score_arr
        return base_array

    def _score_winning_sets(self):
        if self.pwh.get_num_tiles_in_hand() < 14:
            return []
        # if used to make a set, used=True.
        # If used with a set that has used=True, excluded = True.
        # If attempting to use with a set that is excluded, don't.
        # Go through conditions in reverse order so exclusionary rule skips cheap fans.
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

            hand_dict["point_conditions"][77] = nine_gates(tilesets["pungs"], tilesets["kongs"],
                                                           tilesets["chows"], hand_dict["pair"],
                                                           self.pwh.num_suits_used, self.pwh.get_num_honor_tiles())

            hand_dict["point_conditions"][76] = all_green(tilesets["pungs"], tilesets["kongs"],
                                                          tilesets["chows"], hand_dict["pair"])

            hand_dict["point_conditions"][75] = big_three_dragons(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][74] = big_four_winds(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][73] = pure_terminal_chows(tilesets["chows"],
                                                                    self.pwh.num_suits_used,
                                                                    self.pwh.get_num_honor_tiles(), hand_dict["pair"])

            hand_dict["point_conditions"][72] = four_concealed_pungs(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][71] = little_three_dragons(tilesets["pungs"], tilesets["kongs"],
                                                                     hand_dict["pair"])

            hand_dict["point_conditions"][70] = little_four_winds(tilesets["pungs"],
                                                                  tilesets["kongs"], hand_dict["pair"])

            # ALL HONORS
            hand_dict["point_conditions"][69] = 1 if (self.pwh.num_suits_used == 0) else 0

            hand_dict["point_conditions"][68] = all_terminals(tilesets["chows"], tilesets["pungs"],
                                                              tilesets["kongs"], hand_dict["pair"])

            hand_dict["point_conditions"][67] = four_pure_shifted_pungs(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][66] = quad_chow(tilesets["chows"])

            hand_dict["point_conditions"][65] = all_terminals_and_honors(tilesets["chows"], tilesets["pungs"],
                                                                         tilesets["kongs"], hand_dict["pair"])

            hand_dict["point_conditions"][64] = three_kongs(tilesets["kongs"])

            hand_dict["point_conditions"][63] = four_shifted_chows(tilesets["chows"])

            hand_dict["point_conditions"][62] = lower_tiles(tilesets["chows"], tilesets["pungs"],
                                                            tilesets["kongs"], hand_dict["pair"])

            hand_dict["point_conditions"][61] = middle_tiles(tilesets["chows"], tilesets["pungs"],
                                                             tilesets["kongs"], hand_dict["pair"])

            hand_dict["point_conditions"][60] = upper_tiles(tilesets["chows"], tilesets["pungs"],
                                                            tilesets["kongs"], hand_dict["pair"])

            hand_dict["point_conditions"][59] = pure_shifted_pungs(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][58] = pure_triple_chow(tilesets["chows"])

            # 58. FULL FLUSH
            hand_dict["point_conditions"][57] = 1 if (self.pwh.num_suits_used == 1 and
                                                      self.pwh.get_num_honor_tiles() == 0) else 0

            hand_dict["point_conditions"][56] = all_even_pungs(tilesets["pungs"], tilesets["kongs"], hand_dict["pair"])

            # 56. Greater honors + knitted, computed separately
            # 55. Seven pairs, computed separately

            hand_dict["point_conditions"][53] = three_concealed_pungs(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][52] = triple_pung(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][51] = all_fives(tilesets["chows"], tilesets["pungs"],
                                                          tilesets["kongs"], hand_dict["pair"])

            hand_dict["point_conditions"][50] = pure_shifted_chows(tilesets["chows"])

            hand_dict["point_conditions"][49] = three_suited_terminal_chows(tilesets["chows"], hand_dict["pair"])

            hand_dict["point_conditions"][48] = pure_straight(tilesets["chows"])

            hand_dict["point_conditions"][47] = big_three_winds(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][46] = lower_four(tilesets["pungs"], tilesets["kongs"],
                                                           tilesets["chows"], hand_dict["pair"])

            hand_dict["point_conditions"][45] = upper_four(tilesets["pungs"], tilesets["kongs"],
                                                           tilesets["chows"], hand_dict["pair"])

            hand_dict["point_conditions"][44] = 1 if hand_dict["knitted_straight"] else 0

            # 44. Lesser Honors + Knitted Tiles, computed separately

            # 43. CHICKEN HAND WOULD GO HERE BUT YOU NEED TO CHECK SPECIAL CASES

            # 42. KONG ROB
            hand_dict["point_conditions"][41] = 1 if self.robbed_kong else 0

            # 41. REPLACEMENT WIN
            hand_dict["point_conditions"][40] = 1 if self.win_on_replacement else 0

            # 40. LAST TILE CLAIM
            hand_dict["point_conditions"][39] = 1 if self.drew_last_tile and not self.pwh.self_drawn_final_tile else 0

            # 39. LAST TILE DRAW
            hand_dict["point_conditions"][38] = 1 if self.drew_last_tile and self.pwh.self_drawn_final_tile else 0

            hand_dict["point_conditions"][37] = two_concealed_kongs(tilesets["kongs"])

            hand_dict["point_conditions"][36] = mixed_shifted_pungs(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][35] = mixed_triple_chow(tilesets["chows"])

            hand_dict["point_conditions"][34] = reversible_tiles(tilesets["chows"], tilesets["pungs"],
                                                                 tilesets["kongs"], hand_dict["pair"])

            hand_dict["point_conditions"][33] = mixed_straight(tilesets["chows"])

            hand_dict["point_conditions"][32] = two_dragon_pungs(tilesets["pungs"], tilesets["kongs"])

            # 32. MELDED HAND
            if not self.pwh.self_drawn_final_tile and self.pwh.get_num_revealed_sets() == 4 and self.pwh.single_wait:
                hand_dict["point_conditions"][31] = 1

            # 31. ALL TYPES
            hand_dict["point_conditions"][30] = 1 if (self.pwh.get_num_dragons() > 0 and self.pwh.get_num_winds() > 0
                                                      and self.pwh.num_suits_used == 3) else 0

            hand_dict["point_conditions"][29] = mixed_shifted_chow(tilesets["chows"])

            # 29. HALF FLUSH
            hand_dict["point_conditions"][28] = 1 if (self.pwh.num_suits_used == 1 and
                                                      self.pwh.get_num_honor_tiles() > 0) else 0

            # 28. ALL PUNGS
            hand_dict["point_conditions"][27] = 1 if (len(tilesets["chows"]) == 0) else 0

            # 27. LAST TILE
            hand_dict["point_conditions"][26] = 1 if self.last_tile_of_its_kind else 0

            hand_dict["point_conditions"][25] = two_melded_kongs(tilesets["kongs"])

            # 25. FULLY CONCEALED SELF DRAWN
            if self.pwh.self_drawn_final_tile and self.pwh.is_fully_concealed():
                hand_dict["point_conditions"][24] = 1

            hand_dict["point_conditions"][23] = outside_hand(tilesets["chows"], tilesets["pungs"],
                                                             tilesets["kongs"], hand_dict["pair"])

            hand_dict["point_conditions"][22] = all_simples(tilesets["chows"], tilesets["pungs"],
                                                            tilesets["kongs"], hand_dict["pair"])

            hand_dict["point_conditions"][21] = one_concealed_kong(tilesets["kongs"])

            hand_dict["point_conditions"][20] = two_concealed_pungs(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][19] = mixed_double_pung(tilesets["pungs"], tilesets["kongs"])

            hand_dict["point_conditions"][18] = tile_hog(tilesets["pungs"], tilesets["kongs"],
                                                         hand_dict["pair"], tilesets["chows"])

            hand_dict["point_conditions"][17] = all_chow_no_honors(tilesets["pungs"], tilesets["kongs"],
                                                                   hand_dict["pair"])

            # 17. CONCEALED HAND DISCARD WIN
            if self.pwh.is_fully_concealed() and not self.pwh.self_drawn_final_tile:
                hand_dict["point_conditions"][16] = 1

            hand_dict["point_conditions"][15] = seat_wind_pung(tilesets["pungs"], tilesets["kongs"],
                                                               self.tileset_format_seat_wind)

            hand_dict["point_conditions"][14] = round_wind_pung(tilesets["pungs"], tilesets["kongs"], self.tileset_format_round_wind)

            hand_dict["point_conditions"][13] = dragon_pung(tilesets["pungs"], tilesets["kongs"])

            # 13. SINGLE WAIT
            if self.pwh.single_wait:
                hand_dict["point_conditions"][12] = 1

            # 12. CLOSED WAIT
            if self.pwh.closed_wait:
                hand_dict["point_conditions"][11] = 1

            # 11. EDGE WAIT
            if self.pwh.edge_wait:
                hand_dict["point_conditions"][10] = 1
            # 10. SPACE FOR FLOWERS???

            # 09. SELF DRAWN
            if self.pwh.self_drawn_final_tile:
                hand_dict["point_conditions"][8] = 1

            # 08. NO HONOR TILES
            hand_dict["point_conditions"][7] = 1 if (self.pwh.get_num_honor_tiles() == 0) else 0

            # 07. VOIDED SUIT
            hand_dict["point_conditions"][6] = 1 if (self.pwh.num_suits_used <= 2) else 0

            hand_dict["point_conditions"][5] = melded_kong(tilesets["kongs"])

            hand_dict["point_conditions"][4] = terminal_non_dragon_honor_pung(tilesets["pungs"], tilesets["kongs"],
                                                                              self.tileset_format_seat_wind,
                                                                              self.tileset_format_round_wind)

            hand_dict["point_conditions"][3] = two_terminal_chows(tilesets["chows"])

            hand_dict["point_conditions"][2] = short_straight(tilesets["chows"])

            hand_dict["point_conditions"][1] = mixed_double_chow(tilesets["chows"])

            hand_dict["point_conditions"][0] = pure_double_chow(tilesets["chows"])

        base_array = [0] * len(MahjongHands.get_hand_titles())
        sorted_hands = sorted(self.pwh.four_set_pair_hands, key=lambda x: sum(x["point_conditions"]))
        if len(sorted_hands) > 0:
            max_score_array = sorted_hands[0]["point_conditions"]
        else:
            max_score_array = base_array
        return max_score_array



