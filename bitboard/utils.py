from typing import Generator

import numpy as np


def get_main_diagonal_coords(i: int, j: int) -> list[tuple[int, int]]:
    i_start = max(0, i - j)
    j_start = max(0, j - i)
    i_end = 4 - j_start
    j_end = 4 - i_start
    return list(zip(
        range(i_start, i_end + 1),
        range(j_start, j_end + 1),
    ))


def get_secondary_diagonal_coords(i: int, j: int) -> list[tuple[int, int]]:
    i_end = j_end = min(4, i + j)
    i_start = j_start = max(0, i + j - 4)
    return list(zip(
        range(i_end, i_start - 1, -1),
        range(j_start, j_end + 1),
    ))


def get_binary_str_range(n: int, zfill: int = None) -> Generator[str, None, None]:
    if zfill is None:
        zfill = len(bin(n - 1)[2:])
    for i in range(n):
        b = bin(i)[2:].zfill(zfill)
        yield b


def get_attack_mask_str(
        figure_mask: str,  # str of 1 or 0, where 1 - cell is occupied
        attacker_i: int,
        go_left: bool,
        go_right: bool,
) -> str:
    assert figure_mask[attacker_i] == "1"

    attack_mask_str = list("00000")
    if go_left:
        for l in range(attacker_i - 1, -1, -1):  # searching left non-obstructed attacks
            attack_mask_str[l] = "1"
            if figure_mask[l] == "1":
                break
    if go_right:
        for r in range(attacker_i + 1, len(figure_mask)):  # searching right non-obstructed attacks
            attack_mask_str[r] = "1"
            if figure_mask[r] == "1":
                break
    attack_mask_str = "".join(attack_mask_str)
    return attack_mask_str


class Bitarray5x5:
    @staticmethod
    def shift_right(n: int):
        return (n >> 1) & 0b01111_01111_01111_01111_01111

    @staticmethod
    def shift_left(n: int):
        return (n << 1) & 0b11110_11110_11110_11110_11110

    @staticmethod
    def shift_up(n: int):
        return n << 5 & 0b11111_11111_11111_11111_11111

    @staticmethod
    def shift_down(n: int):
        return n >> 5

    @staticmethod
    def repeat_shift(n: int, shift_func, count: int):
        for _ in range(count):
            n = shift_func(n)
        return n

    @staticmethod
    def print(n: int):
        bin_str = bin(n)[2:]
        bin_str = bin_str.zfill(25)  # fill leading zeros if len < 25
        for i in range(5):
            print(bin_str[i * 5: (i + 1) * 5])

    @staticmethod
    def row_to_column(row: int, col_j: int) -> int:
        """
        Rotates :row (5 bits integer) 90 degrees to 5x5 bitarray column placed at index :col_j
        Lowest bit in column becomes most-right, highest - most-left
        """
        col = 0
        col |= row << (5 * 4 - col_j + 0) & (1 << (4 - col_j + 5 * 4))
        col |= row << (5 * 3 - col_j + 1) & (1 << (4 - col_j + 5 * 3))
        col |= row << (5 * 2 - col_j + 2) & (1 << (4 - col_j + 5 * 2))
        col |= row << (5 * 1 - col_j + 3) & (1 << (4 - col_j + 5 * 1))
        col |= row << (5 * 0 - col_j + 4) & (1 << (4 - col_j + 5 * 0))
        return col & COLS[col_j]

    @staticmethod
    def get_column(board: int, col_j: int) -> int:
        row = 0
        row |= board >> (5 * 4 - col_j + 0) & 16
        row |= board >> (5 * 3 - col_j + 1) & 8
        row |= board >> (5 * 2 - col_j + 2) & 4
        row |= board >> (5 * 1 - col_j + 3) & 2
        row |= board >> (5 * 0 - col_j + 4) & 1
        return row & ROWS[-1]

    @staticmethod
    def get_row(board: int, row_i: int) -> int:
        return (board >> (5 * (4 - row_i))) & ROWS[-1]

    @staticmethod
    def main_diagonal_to_row(board: int, i: int, j: int):
        row = 0
        bits_count = 0
        for i, j in reversed(get_main_diagonal_coords(i, j)):
            row |= (board & (i * 5 + 4 - j)) >> (4 - j - bits_count + 5 * (4 - i))
            bits_count += 1
        return row

    @staticmethod
    def secondary_diagonal_to_row(board: int, i: int, j: int):
        row = 0
        bits_count = 0
        for i, j in get_secondary_diagonal_coords(i, j):
            row |= (board & (i * 5 + 4 - j)) >> (4 - j - bits_count + 5 * (4 - i))
            bits_count += 1
        return row


COLS = [
    0b10000_10000_10000_10000_10000,
    0b01000_01000_01000_01000_01000,
    0b00100_00100_00100_00100_00100,
    0b00010_00010_00010_00010_00010,
    0b00001_00001_00001_00001_00001,
]
ROWS = [
    0b11111_00000_00000_00000_00000,
    0b00000_11111_00000_00000_00000,
    0b00000_00000_11111_00000_00000,
    0b00000_00000_00000_11111_00000,
    0b00000_00000_00000_00000_11111,
]

DIAGONAL_MAIN = 0b10000_01000_00100_00010_00001
DIAGONAL_SECONDARY = 0b00001_00010_00100_01000_10000

MAIN_DIAGONALS = [  # (j - i + 4)
    Bitarray5x5.repeat_shift(DIAGONAL_MAIN, Bitarray5x5.shift_down, 4),
    Bitarray5x5.repeat_shift(DIAGONAL_MAIN, Bitarray5x5.shift_down, 3),
    Bitarray5x5.repeat_shift(DIAGONAL_MAIN, Bitarray5x5.shift_down, 2),
    Bitarray5x5.repeat_shift(DIAGONAL_MAIN, Bitarray5x5.shift_down, 1),
    DIAGONAL_MAIN,
    Bitarray5x5.repeat_shift(DIAGONAL_MAIN, Bitarray5x5.shift_right, 1),
    Bitarray5x5.repeat_shift(DIAGONAL_MAIN, Bitarray5x5.shift_right, 2),
    Bitarray5x5.repeat_shift(DIAGONAL_MAIN, Bitarray5x5.shift_right, 3),
    Bitarray5x5.repeat_shift(DIAGONAL_MAIN, Bitarray5x5.shift_right, 4),
]

SECONDARY_DIAGONALS = [  # (j + i)
    Bitarray5x5.repeat_shift(DIAGONAL_SECONDARY, Bitarray5x5.shift_left, 4),
    Bitarray5x5.repeat_shift(DIAGONAL_SECONDARY, Bitarray5x5.shift_left, 3),
    Bitarray5x5.repeat_shift(DIAGONAL_SECONDARY, Bitarray5x5.shift_left, 2),
    Bitarray5x5.repeat_shift(DIAGONAL_SECONDARY, Bitarray5x5.shift_left, 1),
    DIAGONAL_SECONDARY,
    Bitarray5x5.repeat_shift(DIAGONAL_SECONDARY, Bitarray5x5.shift_down, 1),
    Bitarray5x5.repeat_shift(DIAGONAL_SECONDARY, Bitarray5x5.shift_down, 2),
    Bitarray5x5.repeat_shift(DIAGONAL_SECONDARY, Bitarray5x5.shift_down, 3),
    Bitarray5x5.repeat_shift(DIAGONAL_SECONDARY, Bitarray5x5.shift_down, 4),
]

CELLS = [[None for _ in range(5)] for _ in range(5)]
for i in range(5):
    for j in range(5):
        CELLS[i][j] = 1 << ((5 - i - 1) * 5 + (5 - j - 1))

"""
For each rook position in file/rank (5 cells) 
and for each position in other 4 cells (2^4 = 16 combinations) outputs mask of attacks
first dimension - rook position: 1, 2, 4, 8 or 16
second dimension - position of all figures in line, including rook
output - mask, where 1 - cell can be attacked by rook
"""
# ROOK_ATTACKS = np.zeros(shape=(32, 32), dtype=np.uint32)
# for pos in range(5):
#     rook_mask = 1 << pos
#     rook_str_i = 4 - pos
#     for i in range(16):  # iterating all positions of figures in line
#         b = bin(i)[2:].zfill(4)
#         figures_mask_str = b[:4 - pos] + "1" + b[4 - pos:]  # adding rook to line
#         figures_mask = int(figures_mask_str, 2)
#         attack_mask_str = get_attack_mask_str(figures_mask_str, rook_str_i, go_left=True, go_right=True)
#         attack_mask = int(attack_mask_str, 2)
#         # print(bin(rook_pos)[2:].zfill(5), figures_pos_str, attack_mask_str)
#         ROOK_ATTACKS[rook_mask, figures_mask] =  attack_mask

ROOK_VERTICAL_ATTACKS = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
ROOK_HORIZONTAL_ATTACKS = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
for i in range(5):
    for j in range(5):
        # Adding vertical attacks
        for figure_mask_str in get_binary_str_range(32):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[i] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_str = get_attack_mask_str(figure_mask_str, i, go_left=True, go_right=True)
            board_attack_mask_str = [["0" for _ in range(5)] for _ in range(5)]
            for cell_i, attack_status in zip(range(5), attack_mask_str):
                board_attack_mask_str[cell_i][j] = attack_status
            attack_mask = int("".join(["".join(row) for row in board_attack_mask_str]), 2)
            ROOK_VERTICAL_ATTACKS[i, j, figure_mask] = attack_mask

        # Adding horizontal attacks
        for figure_mask_str in get_binary_str_range(32):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[j] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_str = get_attack_mask_str(figure_mask_str, j, go_left=True, go_right=True)
            board_attack_mask_str = [["0" for _ in range(5)] for _ in range(5)]
            for cell_j, attack_status in zip(range(5), attack_mask_str):
                board_attack_mask_str[i][cell_j] = attack_status
            attack_mask = int("".join(["".join(row) for row in board_attack_mask_str]), 2)
            ROOK_HORIZONTAL_ATTACKS[i, j, figure_mask] = attack_mask

"""
dim0 - i
dim1 - j
dim2 - diagonal figure positions represented as 5bit row
"""
MAIN_DIAGONAL_ATTACKS = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
for i in range(5):
    for j in range(5):
        diagonal_cells = get_main_diagonal_coords(i, j)
        bishop_i = diagonal_cells.index((i, j))
        for figure_mask_str in get_binary_str_range(2 ** len(diagonal_cells)):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[bishop_i] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_str = get_attack_mask_str(figure_mask_str, bishop_i, go_left=True, go_right=True)
            board_attack_mask_str = [["0" for _ in range(5)] for _ in range(5)]
            for (cell_i, cell_j), attack_status in zip(diagonal_cells, attack_mask_str):
                board_attack_mask_str[cell_i][cell_j] = attack_status
            attack_mask = int("".join(["".join(row) for row in board_attack_mask_str]), 2)
            MAIN_DIAGONAL_ATTACKS[i, j, figure_mask] = attack_mask

"""
dim0 - i
dim1 - j
dim2 - diagonal figure positions represented as 5bit row
"""
SECONDARY_DIAGONAL_ATTACKS = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
for i in range(5):
    for j in range(5):
        diagonal_cells = get_secondary_diagonal_coords(i, j)
        bishop_i = diagonal_cells.index((i, j))
        for figure_mask_str in get_binary_str_range(2 ** len(diagonal_cells)):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[bishop_i] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_str = get_attack_mask_str(figure_mask_str, bishop_i, go_left=True, go_right=True)
            board_attack_mask_str = [["0" for _ in range(5)] for _ in range(5)]
            for (cell_i, cell_j), attack_status in zip(diagonal_cells, attack_mask_str):
                board_attack_mask_str[cell_i][cell_j] = attack_status
            attack_mask = int("".join(["".join(row) for row in board_attack_mask_str]), 2)
            SECONDARY_DIAGONAL_ATTACKS[i, j, figure_mask] = attack_mask

"""
Same masks used for rook but separated in two directions: 
left and right (up and down when we rotate attack mask by 90 degrees)
"""
LANCE_ATTACKS_BLACK = np.zeros(shape=(32, 32), dtype=np.uint32)
LANCE_ATTACKS_WHITE = np.zeros(shape=(32, 32), dtype=np.uint32)
for pos in range(5):
    lance_mask = 1 << pos
    lance_str_i = 4 - pos
    for i in range(16):  # iterating all positions of figures in line
        b = bin(i)[2:].zfill(4)
        figures_mask_str = b[:4 - pos] + "1" + b[4 - pos:]  # adding rook to line
        figures_mask = int(figures_mask_str, 2)
        attack_mask_black_str = get_attack_mask_str(figures_mask_str, lance_str_i, go_left=True, go_right=False)
        attack_mask_white_str = get_attack_mask_str(figures_mask_str, lance_str_i, go_left=False, go_right=True)
        attack_mask_black = int(attack_mask_black_str, 2)
        attack_mask_white = int(attack_mask_white_str, 2)
        LANCE_ATTACKS_BLACK[lance_mask, figures_mask] = attack_mask_black
        LANCE_ATTACKS_WHITE[lance_mask, figures_mask] = attack_mask_white

LANCE_ATTACKS_BLACK = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
LANCE_ATTACKS_WHITE = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
for i in range(5):
    for j in range(5):
        for figure_mask_str in get_binary_str_range(32):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[i] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_black_str = get_attack_mask_str(figure_mask_str, i, go_left=True, go_right=False)
            attack_mask_white_str = get_attack_mask_str(figure_mask_str, i, go_left=False, go_right=True)
            board_attack_mask_black_str = [["0" for _ in range(5)] for _ in range(5)]
            board_attack_mask_white_str = [["0" for _ in range(5)] for _ in range(5)]
            for cell_i, attack_status in zip(range(5), attack_mask_black_str):
                board_attack_mask_black_str[cell_i][j] = attack_status
            for cell_i, attack_status in zip(range(5), attack_mask_white_str):
                board_attack_mask_white_str[cell_i][j] = attack_status
            attack_mask_black = int("".join(["".join(row) for row in board_attack_mask_black_str]), 2)
            attack_mask_white = int("".join(["".join(row) for row in board_attack_mask_white_str]), 2)
            LANCE_ATTACKS_BLACK[i, j, figure_mask] = attack_mask_black
            LANCE_ATTACKS_WHITE[i, j, figure_mask] = attack_mask_white

GOLD_ATTACKS_BLACK_PATTERN = 0b00000_01110_01010_00100_00000
GOLD_ATTACKS_WHITE_PATTERN = 0b00000_00100_01010_01110_00000
SILVER_ATTACKS_BLACK_PATTERN = 0b00000_01110_00000_01010_00000
SILVER_ATTACKS_WHITE_PATTERN = 0b00000_01010_00000_01110_00000
KING_ATTACKS_PATTERN = 0b00000_01110_01010_01110_00000
KNIGHT_ATTACKS_BLACK_PATTERN = 0b01010_00000_00000_00000_00000
KNIGHT_ATTACKS_WHITE_PATTERN = 0b00000_00000_00000_00000_01010
PAWN_ATTACKS_BLACK_PATTERN = 0b00000_00100_00000_00000_00000
PAWN_ATTACKS_WHITE_PATTERN = 0b00000_00000_00000_00100_00000


def move_attack_to(attack_mask: int, i: int, j: int) -> int:
    di = 2 - i
    dj = 2 - j
    if di >= 0:
        attack_mask = Bitarray5x5.repeat_shift(attack_mask, Bitarray5x5.shift_up, di)
    else:
        attack_mask = Bitarray5x5.repeat_shift(attack_mask, Bitarray5x5.shift_down, -di)
    if dj >= 0:
        attack_mask = Bitarray5x5.repeat_shift(attack_mask, Bitarray5x5.shift_left, dj)
    else:
        attack_mask = Bitarray5x5.repeat_shift(attack_mask, Bitarray5x5.shift_right, -dj)
    return attack_mask


GOLD_ATTACKS_BLACK = np.zeros(shape=(5, 5), dtype=np.uint32)
GOLD_ATTACKS_WHITE = np.zeros(shape=(5, 5), dtype=np.uint32)
SILVER_ATTACKS_BLACK = np.zeros(shape=(5, 5), dtype=np.uint32)
SILVER_ATTACKS_WHITE = np.zeros(shape=(5, 5), dtype=np.uint32)
KING_ATTACKS = np.zeros(shape=(5, 5), dtype=np.uint32)
KNIGHT_ATTACKS_BLACK = np.zeros(shape=(5, 5), dtype=np.uint32)
KNIGHT_ATTACKS_WHITE = np.zeros(shape=(5, 5), dtype=np.uint32)
PAWN_ATTACKS_BLACK = np.zeros(shape=(5, 5), dtype=np.uint32)
PAWN_ATTACKS_WHITE = np.zeros(shape=(5, 5), dtype=np.uint32)

for i in range(5):
    for j in range(5):
        GOLD_ATTACKS_BLACK[i, j] = move_attack_to(GOLD_ATTACKS_BLACK_PATTERN, i, j)
        GOLD_ATTACKS_WHITE[i, j] = move_attack_to(GOLD_ATTACKS_WHITE_PATTERN, i, j)
        SILVER_ATTACKS_BLACK[i, j] = move_attack_to(SILVER_ATTACKS_BLACK_PATTERN, i, j)
        SILVER_ATTACKS_WHITE[i, j] = move_attack_to(SILVER_ATTACKS_WHITE_PATTERN, i, j)
        KING_ATTACKS[i, j] = move_attack_to(KING_ATTACKS_PATTERN, i, j)
        KNIGHT_ATTACKS_BLACK[i, j] = move_attack_to(KNIGHT_ATTACKS_BLACK_PATTERN, i, j)
        KNIGHT_ATTACKS_WHITE[i, j] = move_attack_to(KNIGHT_ATTACKS_WHITE_PATTERN, i, j)
        PAWN_ATTACKS_BLACK[i, j] = move_attack_to(PAWN_ATTACKS_BLACK_PATTERN, i, j)
        PAWN_ATTACKS_WHITE[i, j] = move_attack_to(PAWN_ATTACKS_WHITE_PATTERN, i, j)


def get_rook_attacks(figure_mask: int, rook_i: int, rook_j: int) -> int:
    col_figure_mask = Bitarray5x5.get_column(figure_mask, rook_j)
    row_figure_mask = Bitarray5x5.get_row(figure_mask, rook_i)
    vert_attack_mask = ROOK_VERTICAL_ATTACKS[rook_i, rook_j, col_figure_mask]
    horiz_attack_mask = ROOK_HORIZONTAL_ATTACKS[rook_i, rook_j, row_figure_mask]
    return vert_attack_mask | horiz_attack_mask


def get_bishop_attacks(figure_mask: int, bishop_i: int, bishop_j: int) -> int:
    main_diag_figure_mask = Bitarray5x5.main_diagonal_to_row(figure_mask, bishop_i, bishop_j)
    sec_diag_figure_mask = Bitarray5x5.secondary_diagonal_to_row(figure_mask, bishop_i, bishop_j)
    main_attack_mask = MAIN_DIAGONAL_ATTACKS[bishop_i, bishop_j, main_diag_figure_mask]
    sec_attack_mask = SECONDARY_DIAGONAL_ATTACKS[bishop_i, bishop_j, sec_diag_figure_mask]
    return main_attack_mask | sec_attack_mask


def get_black_lance_attacks(figure_mask: int, lance_i: int, lance_j: int) -> int:
    col_figure_mask = Bitarray5x5.get_column(figure_mask, lance_j)
    return LANCE_ATTACKS_BLACK[lance_i, lance_j, col_figure_mask]


def get_white_lance_attacks(figure_mask: int, lance_i: int, lance_j: int) -> int:
    col_figure_mask = Bitarray5x5.get_column(figure_mask, lance_j)
    return LANCE_ATTACKS_WHITE[lance_i, lance_j, col_figure_mask]
