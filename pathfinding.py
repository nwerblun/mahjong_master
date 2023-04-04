from game import *
from score_calculator import Calculator


class Pathfinder:
    def __init__(self):
        self.calc = Calculator()
        self.start_hand = None

    def set_calc(self, concealed, revealed, final, self_drawn_final, concealed_kongs, revealed_kongs,
                 round_wind, seat_wind_cv):
        self.calc.set_hand(concealed, revealed, final, self_drawn_final, concealed_kongs,
                           revealed_kongs, round_wind, seat_wind_cv)
        self.start_hand = self.calc.pwh

