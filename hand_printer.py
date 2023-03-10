import pdb

from hands import MahjongHands
import os

import pdb
def print_hands():
    num_categories = 4
    term_width = os.get_terminal_size()
    num_cols = term_width[0]
    # num_rows = term_width[1]
    headers = ["Name", "Point Value", "Categories", "Notes"]
    longest_header = max(map(lambda x: len(x), headers))
    # + 2 + len(headers)-1 because of | on left and right and | in between each header
    min_len_required = (num_categories * longest_header) + 2 + (len(headers) - 1)
    if (num_cols - min_len_required) < 0:
        print("Can't print, terminal size too small")
        return
    print("-" * num_cols)
    # 2 + len(headers) - 1 = number of | separators
    space_per_header = (num_cols - 2 - (len(headers) - 1)) // num_categories
    for i in range(len(headers)):
        text_len = len(headers[i])
        spaces_per_side = (space_per_header - text_len) // 2
        headers[i] = (" " * spaces_per_side) + headers[i] + (" " * spaces_per_side)
    str_to_print = "|" + "|".join(headers) + "|"
    # Add any remaining space due to // to last header. Add 1 extra space to compensate for using [:-1]
    str_to_print = str_to_print[:-1] + (" " * (num_cols - len(str_to_print))) + "|"
    print(str_to_print)
    print("-" * num_cols)

    # Remove the 5 |'s and divide into sections.
    for key in MahjongHands.hands_info.keys():
        name = key[:]
        point_value = str(MahjongHands.hands_info[key][0])
        category_text = MahjongHands.hands_info[key][1][:]
        note_text = MahjongHands.hands_info[key][2][:]





print_hands()
