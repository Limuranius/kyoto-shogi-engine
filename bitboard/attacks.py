from typing import Generator

import numpy as np
from .bitarray5x5 import Bitarray5x5, VectorizableInt
from . import bitarray5x5



def _get_binary_str_range(n: int, zfill: int = None) -> Generator[str, None, None]:
    if zfill is None:
        zfill = len(bin(n - 1)[2:])
    for i in range(n):
        b = bin(i)[2:].zfill(zfill)
        yield b


def _get_line_attack_mask_str(
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



"""
For each rook position in file/rank (5 cells) 
and for each position in other 4 cells (2^4 = 16 combinations) outputs mask of attacks
first dimension - rook position: 1, 2, 4, 8 or 16
second dimension - position of all figures in line, including rook
output - mask, where 1 - cell can be attacked by rook
"""
ROOK_VERTICAL_ATTACKS = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
ROOK_HORIZONTAL_ATTACKS = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
for i in range(5):
    for j in range(5):
        # Adding vertical attacks
        for figure_mask_str in _get_binary_str_range(32):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[i] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_str = _get_line_attack_mask_str(figure_mask_str, i, go_left=True, go_right=True)
            board_attack_mask_str = [["0" for _ in range(5)] for _ in range(5)]
            for cell_i, attack_status in zip(range(5), attack_mask_str):
                board_attack_mask_str[cell_i][j] = attack_status
            attack_mask = int("".join(["".join(row) for row in board_attack_mask_str]), 2)
            ROOK_VERTICAL_ATTACKS[i, j, figure_mask] = attack_mask

        # Adding horizontal attacks
        for figure_mask_str in _get_binary_str_range(32):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[j] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_str = _get_line_attack_mask_str(figure_mask_str, j, go_left=True, go_right=True)
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
        diagonal_cells = bitarray5x5.get_main_diagonal_coords(i, j)
        bishop_i = diagonal_cells.index((i, j))
        for figure_mask_str in _get_binary_str_range(2 ** len(diagonal_cells)):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[bishop_i] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_str = _get_line_attack_mask_str(figure_mask_str, bishop_i, go_left=True, go_right=True)
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
        diagonal_cells = bitarray5x5.get_secondary_diagonal_coords(i, j)
        bishop_i = diagonal_cells.index((i, j))
        for figure_mask_str in _get_binary_str_range(2 ** len(diagonal_cells)):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[bishop_i] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_str = _get_line_attack_mask_str(figure_mask_str, bishop_i, go_left=True, go_right=True)
            board_attack_mask_str = [["0" for _ in range(5)] for _ in range(5)]
            for (cell_i, cell_j), attack_status in zip(diagonal_cells, attack_mask_str):
                board_attack_mask_str[cell_i][cell_j] = attack_status
            attack_mask = int("".join(["".join(row) for row in board_attack_mask_str]), 2)
            SECONDARY_DIAGONAL_ATTACKS[i, j, figure_mask] = attack_mask

"""
Same masks used for rook but separated in two directions: 
left and right (up and down when we rotate attack mask by 90 degrees)
"""
LANCE_ATTACKS_BLACK = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
LANCE_ATTACKS_WHITE = np.zeros(shape=(5, 5, 32), dtype=np.uint32)
for i in range(5):
    for j in range(5):
        for figure_mask_str in _get_binary_str_range(32):
            figure_mask_str = list(figure_mask_str)
            figure_mask_str[i] = "1"
            figure_mask_str = "".join(figure_mask_str)
            figure_mask = int(figure_mask_str, 2)
            attack_mask_black_str = _get_line_attack_mask_str(figure_mask_str, i, go_left=True, go_right=False)
            attack_mask_white_str = _get_line_attack_mask_str(figure_mask_str, i, go_left=False, go_right=True)
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


def _move_attack_to(attack_mask: int, i: int, j: int) -> int:
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
        GOLD_ATTACKS_BLACK[i, j] = _move_attack_to(GOLD_ATTACKS_BLACK_PATTERN, i, j)
        GOLD_ATTACKS_WHITE[i, j] = _move_attack_to(GOLD_ATTACKS_WHITE_PATTERN, i, j)
        SILVER_ATTACKS_BLACK[i, j] = _move_attack_to(SILVER_ATTACKS_BLACK_PATTERN, i, j)
        SILVER_ATTACKS_WHITE[i, j] = _move_attack_to(SILVER_ATTACKS_WHITE_PATTERN, i, j)
        KING_ATTACKS[i, j] = _move_attack_to(KING_ATTACKS_PATTERN, i, j)
        KNIGHT_ATTACKS_BLACK[i, j] = _move_attack_to(KNIGHT_ATTACKS_BLACK_PATTERN, i, j)
        KNIGHT_ATTACKS_WHITE[i, j] = _move_attack_to(KNIGHT_ATTACKS_WHITE_PATTERN, i, j)
        PAWN_ATTACKS_BLACK[i, j] = _move_attack_to(PAWN_ATTACKS_BLACK_PATTERN, i, j)
        PAWN_ATTACKS_WHITE[i, j] = _move_attack_to(PAWN_ATTACKS_WHITE_PATTERN, i, j)


# Flat version of arrays of all attacks ================================

ROOK_VERTICAL_ATTACKS_FLAT = np.zeros(shape=(25, 32), dtype=np.uint32)
ROOK_HORIZONTAL_ATTACKS_FLAT = np.zeros(shape=(25, 32), dtype=np.uint32)
MAIN_DIAGONAL_ATTACKS_FLAT = np.zeros(shape=(25, 32), dtype=np.uint32)
SECONDARY_DIAGONAL_ATTACKS_FLAT = np.zeros(shape=(25, 32), dtype=np.uint32)
LANCE_ATTACKS_BLACK_FLAT = np.zeros(shape=(25, 32), dtype=np.uint32)
LANCE_ATTACKS_WHITE_FLAT = np.zeros(shape=(25, 32), dtype=np.uint32)

GOLD_ATTACKS_BLACK_FLAT = np.zeros(shape=25, dtype=np.uint32)
GOLD_ATTACKS_WHITE_FLAT = np.zeros(shape=25, dtype=np.uint32)
SILVER_ATTACKS_BLACK_FLAT = np.zeros(shape=25, dtype=np.uint32)
SILVER_ATTACKS_WHITE_FLAT = np.zeros(shape=25, dtype=np.uint32)
KING_ATTACKS_FLAT = np.zeros(shape=25, dtype=np.uint32)
KNIGHT_ATTACKS_BLACK_FLAT = np.zeros(shape=25, dtype=np.uint32)
KNIGHT_ATTACKS_WHITE_FLAT = np.zeros(shape=25, dtype=np.uint32)
PAWN_ATTACKS_BLACK_FLAT = np.zeros(shape=25, dtype=np.uint32)
PAWN_ATTACKS_WHITE_FLAT = np.zeros(shape=25, dtype=np.uint32)

for i in range(5):
    for j in range(5):
        ROOK_VERTICAL_ATTACKS_FLAT[bitarray5x5.coord_to_shift(i, j)] = ROOK_VERTICAL_ATTACKS[i, j]
        ROOK_HORIZONTAL_ATTACKS_FLAT[bitarray5x5.coord_to_shift(i, j)] = ROOK_HORIZONTAL_ATTACKS[i, j]
        MAIN_DIAGONAL_ATTACKS_FLAT[bitarray5x5.coord_to_shift(i, j)] = MAIN_DIAGONAL_ATTACKS[i, j]
        SECONDARY_DIAGONAL_ATTACKS_FLAT[bitarray5x5.coord_to_shift(i, j)] = SECONDARY_DIAGONAL_ATTACKS[i, j]
        LANCE_ATTACKS_BLACK_FLAT[bitarray5x5.coord_to_shift(i, j)] = LANCE_ATTACKS_BLACK[i, j]
        LANCE_ATTACKS_WHITE_FLAT[bitarray5x5.coord_to_shift(i, j)] = LANCE_ATTACKS_WHITE[i, j]

        GOLD_ATTACKS_BLACK_FLAT[bitarray5x5.coord_to_shift(i, j)] = GOLD_ATTACKS_BLACK[i, j]
        GOLD_ATTACKS_WHITE_FLAT[bitarray5x5.coord_to_shift(i, j)] = GOLD_ATTACKS_WHITE[i, j]
        SILVER_ATTACKS_BLACK_FLAT[bitarray5x5.coord_to_shift(i, j)] = SILVER_ATTACKS_BLACK[i, j]
        SILVER_ATTACKS_WHITE_FLAT[bitarray5x5.coord_to_shift(i, j)] = SILVER_ATTACKS_WHITE[i, j]
        KING_ATTACKS_FLAT[bitarray5x5.coord_to_shift(i, j)] = KING_ATTACKS[i, j]
        KNIGHT_ATTACKS_BLACK_FLAT[bitarray5x5.coord_to_shift(i, j)] = KNIGHT_ATTACKS_BLACK[i, j]
        KNIGHT_ATTACKS_WHITE_FLAT[bitarray5x5.coord_to_shift(i, j)] = KNIGHT_ATTACKS_WHITE[i, j]
        PAWN_ATTACKS_BLACK_FLAT[bitarray5x5.coord_to_shift(i, j)] = PAWN_ATTACKS_BLACK[i, j]
        PAWN_ATTACKS_WHITE_FLAT[bitarray5x5.coord_to_shift(i, j)] = PAWN_ATTACKS_WHITE[i, j]


def get_rook_attacks(
        figure_mask: VectorizableInt,
        rook_i: VectorizableInt,
        rook_j: VectorizableInt,
) -> VectorizableInt:
    col_figure_mask = Bitarray5x5.get_column(figure_mask, rook_j)
    row_figure_mask = Bitarray5x5.get_row(figure_mask, rook_i)
    vert_attack_mask = ROOK_VERTICAL_ATTACKS[rook_i, rook_j, col_figure_mask]
    horiz_attack_mask = ROOK_HORIZONTAL_ATTACKS[rook_i, rook_j, row_figure_mask]
    return vert_attack_mask | horiz_attack_mask


def get_rook_attacks_from_shift(
        figure_mask: VectorizableInt,
        shift: VectorizableInt,
) -> VectorizableInt:
    col_figure_masks = Bitarray5x5.get_column(figure_mask, bitarray5x5.shift_to_col(shift))
    row_figure_masks = Bitarray5x5.get_row(figure_mask, bitarray5x5.shift_to_row(shift))
    vert_attack_masks = ROOK_VERTICAL_ATTACKS_FLAT[shift, col_figure_masks]
    horiz_attack_masks = ROOK_HORIZONTAL_ATTACKS_FLAT[shift, row_figure_masks]
    return vert_attack_masks | horiz_attack_masks


def get_bishop_attacks(
        figure_mask: VectorizableInt,
        bishop_i: VectorizableInt,
        bishop_j: VectorizableInt,
) -> VectorizableInt:
    shift = bitarray5x5.coord_to_shift(bishop_i, bishop_j)
    main_diag_figure_mask = Bitarray5x5.main_diagonal_to_row(figure_mask, shift)
    sec_diag_figure_mask = Bitarray5x5.secondary_diagonal_to_row(figure_mask, shift)
    main_attack_mask = MAIN_DIAGONAL_ATTACKS[bishop_i, bishop_j, main_diag_figure_mask]
    sec_attack_mask = SECONDARY_DIAGONAL_ATTACKS[bishop_i, bishop_j, sec_diag_figure_mask]
    return main_attack_mask | sec_attack_mask


def get_bishop_attacks_from_shift(
        figure_mask: VectorizableInt,
        shift: VectorizableInt,
) -> VectorizableInt:
    main_diag_figure_mask = Bitarray5x5.main_diagonal_to_row(figure_mask, shift)
    sec_diag_figure_mask = Bitarray5x5.secondary_diagonal_to_row(figure_mask, shift)
    main_attack_mask = MAIN_DIAGONAL_ATTACKS_FLAT[shift, main_diag_figure_mask]
    sec_attack_mask = SECONDARY_DIAGONAL_ATTACKS_FLAT[shift, sec_diag_figure_mask]
    return main_attack_mask | sec_attack_mask


def get_black_lance_attacks(
        figure_mask: VectorizableInt,
        lance_i: VectorizableInt,
        lance_j: VectorizableInt,
) -> VectorizableInt:
    col_figure_mask = Bitarray5x5.get_column(figure_mask, lance_j)
    return LANCE_ATTACKS_BLACK[lance_i, lance_j, col_figure_mask]


def get_black_lance_attacks_from_shift(
        figure_mask: VectorizableInt,
        shift: VectorizableInt,
) -> VectorizableInt:
    col_figure_mask = Bitarray5x5.get_column(figure_mask, bitarray5x5.shift_to_col(shift))
    return LANCE_ATTACKS_BLACK_FLAT[shift, col_figure_mask]



def get_white_lance_attacks(
        figure_mask: VectorizableInt,
        lance_i: VectorizableInt,
        lance_j: VectorizableInt,
) -> VectorizableInt:
    col_figure_mask = Bitarray5x5.get_column(figure_mask, lance_j)
    return LANCE_ATTACKS_WHITE[lance_i, lance_j, col_figure_mask]


def get_white_lance_attacks_from_shift(
        figure_mask: VectorizableInt,
        shift: VectorizableInt,
) -> VectorizableInt:
    col_figure_mask = Bitarray5x5.get_column(figure_mask, bitarray5x5.shift_to_col(shift))
    return LANCE_ATTACKS_WHITE_FLAT[shift, col_figure_mask]
