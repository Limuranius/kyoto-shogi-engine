import numpy as np

from bitboard.utils import get_bishop_attacks, get_rook_attacks
from utils import *

(
    # 5x5 masks of figure positions stored in uint32
    TOKIN_BLACK,
    LANCE_BLACK,
    SILVER_BLACK,
    BISHOP_BLACK,
    KING_BLACK,
    GOLD_BLACK,
    KNIGHT_BLACK,
    PAWN_BLACK,
    ROOK_BLACK,
    TOKIN_WHITE,
    LANCE_WHITE,
    SILVER_WHITE,
    BISHOP_WHITE,
    KING_WHITE,
    GOLD_WHITE,
    KNIGHT_WHITE,
    PAWN_WHITE,
    ROOK_WHITE,

    # Inventory. Stored for both sides in one uint32. Each 4 bits represent count of figures in inventory.
    INVENTORY,

    # Mask of all attacked cells
    ATTACKS_BLACK,
    ATTACKS_WHITE,

    # Mask of all attacked cells, including friendly-fire
    ATTACKS_FF_BLACK,
    ATTACKS_FF_WHITE,

    # Masks of black, white, empty and occupied cells
    IS_BLACK,
    IS_WHITE,
    IS_EMPTY,
    IS_OCCUPIED,

    # black's turn flag. 1 or 0
    IS_BLACK_TURN,
) = range(5)

# How many bits need to be shifted in inventory to get figure count
INVENTORY_SHIFT = np.zeros(18, dtype=int)
INVENTORY_SHIFT[TOKIN_BLACK] = 0
INVENTORY_SHIFT[LANCE_BLACK] = 0
INVENTORY_SHIFT[SILVER_BLACK] = 3
INVENTORY_SHIFT[BISHOP_BLACK] = 3
INVENTORY_SHIFT[KING_BLACK] = 6
INVENTORY_SHIFT[GOLD_BLACK] = 9
INVENTORY_SHIFT[KNIGHT_BLACK] = 9
INVENTORY_SHIFT[PAWN_BLACK] = 12
INVENTORY_SHIFT[ROOK_BLACK] = 12
INVENTORY_SHIFT[TOKIN_WHITE] = 15
INVENTORY_SHIFT[LANCE_WHITE] = 15
INVENTORY_SHIFT[SILVER_WHITE] = 18
INVENTORY_SHIFT[BISHOP_WHITE] = 18
INVENTORY_SHIFT[KING_WHITE] = 21
INVENTORY_SHIFT[GOLD_WHITE] = 24
INVENTORY_SHIFT[KNIGHT_WHITE] = 24
INVENTORY_SHIFT[PAWN_WHITE] = 27
INVENTORY_SHIFT[ROOK_WHITE] = 27
INVENTORY_CLIP = 0b111

# what figure index do we get when we flip figure
FLIP_FIGURE = np.zeros(18, dtype=int)
FLIP_FIGURE[TOKIN_BLACK] = LANCE_BLACK
FLIP_FIGURE[LANCE_BLACK] = TOKIN_BLACK
FLIP_FIGURE[SILVER_BLACK] = BISHOP_BLACK
FLIP_FIGURE[BISHOP_BLACK] = SILVER_BLACK
FLIP_FIGURE[KING_BLACK] = KING_BLACK
FLIP_FIGURE[GOLD_BLACK] = KNIGHT_BLACK
FLIP_FIGURE[KNIGHT_BLACK] = GOLD_BLACK
FLIP_FIGURE[PAWN_BLACK] = ROOK_BLACK
FLIP_FIGURE[ROOK_BLACK] = PAWN_BLACK
FLIP_FIGURE[TOKIN_WHITE] = LANCE_WHITE
FLIP_FIGURE[LANCE_WHITE] = TOKIN_WHITE
FLIP_FIGURE[SILVER_WHITE] = BISHOP_WHITE
FLIP_FIGURE[BISHOP_WHITE] = SILVER_WHITE
FLIP_FIGURE[KING_WHITE] = KING_WHITE
FLIP_FIGURE[GOLD_WHITE] = KNIGHT_WHITE
FLIP_FIGURE[KNIGHT_WHITE] = GOLD_WHITE
FLIP_FIGURE[PAWN_WHITE] = ROOK_WHITE
FLIP_FIGURE[ROOK_WHITE] = PAWN_WHITE


Bitboard = np.ndarray  # Bitboard, stored in uint32 array
PositionBit = int
BitboardIndex = int

FIGURES_WITH_SHORT_ATTACK = {
    TOKIN_BLACK: GOLD_ATTACKS_BLACK,
    SILVER_BLACK: SILVER_ATTACKS_BLACK,
    KING_BLACK: KING_ATTACKS,
    GOLD_BLACK: GOLD_ATTACKS_BLACK,
    KNIGHT_BLACK: KNIGHT_ATTACKS_BLACK,
    PAWN_BLACK: PAWN_ATTACKS_BLACK,
    TOKIN_WHITE: GOLD_ATTACKS_WHITE,
    SILVER_WHITE: SILVER_ATTACKS_WHITE,
    KING_WHITE: KING_ATTACKS,
    GOLD_WHITE: GOLD_ATTACKS_WHITE,
    KNIGHT_WHITE: KNIGHT_ATTACKS_WHITE,
    PAWN_WHITE: PAWN_ATTACKS_WHITE,
}

FIGURES_WITH_LONG_ATTACK = {
    LANCE_BLACK: get_black_lance_attacks,
    BISHOP_BLACK: get_bishop_attacks,
    ROOK_BLACK: get_rook_attacks,
    LANCE_WHITE: get_black_lance_attacks,
    BISHOP_WHITE: get_bishop_attacks,
    ROOK_WHITE: get_rook_attacks,
}

IS_FIGURE_BLACK = [False] * 18
IS_FIGURE_BLACK[TOKIN_BLACK] = True
IS_FIGURE_BLACK[LANCE_BLACK] = True
IS_FIGURE_BLACK[SILVER_BLACK] = True
IS_FIGURE_BLACK[BISHOP_BLACK] = True
IS_FIGURE_BLACK[KING_BLACK] = True
IS_FIGURE_BLACK[GOLD_BLACK] = True
IS_FIGURE_BLACK[KNIGHT_BLACK] = True
IS_FIGURE_BLACK[PAWN_BLACK] = True
IS_FIGURE_BLACK[ROOK_BLACK] = True
IS_FIGURE_BLACK[TOKIN_WHITE] = False
IS_FIGURE_BLACK[LANCE_WHITE] = False
IS_FIGURE_BLACK[SILVER_WHITE] = False
IS_FIGURE_BLACK[BISHOP_WHITE] = False
IS_FIGURE_BLACK[KING_WHITE] = False
IS_FIGURE_BLACK[GOLD_WHITE] = False
IS_FIGURE_BLACK[KNIGHT_WHITE] = False
IS_FIGURE_BLACK[PAWN_WHITE] = False
IS_FIGURE_BLACK[ROOK_WHITE] = False


FIGURE_INDICES = list(range(TOKIN_BLACK, ROOK_WHITE + 1))


def least_significant_bit(n: int):
    return n & -n


def lsb_position(lsb: PositionBit):
    # Position (number of shifts) of least significant bit
    return np.bitwise_count(lsb - 1)


def get_figure_attack_mask(
        figure_index: int,
        figure_mask: int,
        i: int,
        j: int
) -> int:
    if figure_index in FIGURES_WITH_SHORT_ATTACK:
        return FIGURES_WITH_SHORT_ATTACK[figure_index][i, j]
    else:
        return FIGURES_WITH_LONG_ATTACK[figure_index](figure_mask, i, j)


def update_masks(bitboard: Bitboard) -> None:
    """Updates all masks of bitboard according to placement of pieces"""
    black_attacks_mask = 0
    white_attacks_mask = 0

    bitboard[IS_BLACK] = np.bitwise_or(bitboard[TOKIN_BLACK: TOKIN_WHITE])
    bitboard[IS_WHITE] = np.bitwise_or(bitboard[TOKIN_WHITE: ROOK_WHITE + 1])
    bitboard[IS_OCCUPIED] = bitboard[IS_BLACK] | bitboard[IS_WHITE]
    bitboard[IS_EMPTY] = ~bitboard[IS_OCCUPIED]

    for figure_index in FIGURE_INDICES:
        position_mask = bitboard[figure_index]
        while position_mask:
            lsb = least_significant_bit(position_mask)
            pos = lsb_position(lsb)
            i = 4 - pos // 5
            j = 4 - pos % 5
            attack_mask = get_figure_attack_mask(figure_index, bitboard[IS_OCCUPIED], i, j)
            if IS_FIGURE_BLACK[figure_index]:
                black_attacks_mask |= attack_mask  # add attack to total mask
            else:
                white_attacks_mask |= attack_mask
            position_mask ^= lsb  # remove lowest bit

    bitboard[ATTACKS_FF_BLACK] = black_attacks_mask
    bitboard[ATTACKS_FF_WHITE] = white_attacks_mask
    bitboard[ATTACKS_BLACK] = black_attacks_mask & ~bitboard[IS_BLACK]
    bitboard[ATTACKS_WHITE] = white_attacks_mask & ~bitboard[IS_WHITE]


def get_figure_index(
        bitboard: Bitboard,
        cell_bit: PositionBit
) -> BitboardIndex:
    return (bitboard[0: ROOK_WHITE + 1] & cell_bit).argmax()


def get_inventory_count(
        bitboard: Bitboard,
        figure_index: BitboardIndex,
) -> int:
    return (bitboard[INVENTORY] >> INVENTORY_SHIFT[figure_index]) & INVENTORY_CLIP


def increase_inventory_count(
        bitboard: Bitboard,
        figure_index: BitboardIndex,
) -> None:
    bitboard[INVENTORY] += 1 << INVENTORY_SHIFT[figure_index]


def decrease_inventory_count(
        bitboard: Bitboard,
        figure_index: BitboardIndex,
) -> None:
    bitboard[INVENTORY] -= 1 << INVENTORY_SHIFT[figure_index]


def make_move_fast(
        bitboard: Bitboard,
        prev_pos_bit: PositionBit,
        new_pos_bit: PositionBit,
        figure_index: BitboardIndex,
) -> Bitboard:
    """
    prev_pos_bit - bit on the board, where piece was before the move
    new_pos_bit - bit on the board, where piece was after the move

    Assumptions for input (not checked in this function):
        Previous figure position is not empty
        Previous figure color matches current turn
    """
    new_bitboard = bitboard.copy()

    new_bitboard[FLIP_FIGURE[figure_index]] |= new_pos_bit
    new_bitboard[figure_index] ^= prev_pos_bit
    if not (new_bitboard[IS_EMPTY] & new_pos_bit):  # opponent's piece is taken
        taken_figure_index = get_figure_index(new_bitboard, new_pos_bit)
        increase_inventory_count(new_bitboard, taken_figure_index)  # increase inventory count
        new_bitboard[taken_figure_index] ^= new_pos_bit  # remove taken piece
    new_bitboard[IS_BLACK_TURN] = int(not new_bitboard[IS_BLACK_TURN])

    update_masks(new_bitboard)
    return new_bitboard


def make_drop_fast(
        bitboard: Bitboard,
        drop_pos_bit: PositionBit,
        figure_index: BitboardIndex,
) -> Bitboard:
    new_bitboard = bitboard.copy()

    new_bitboard[figure_index] |= drop_pos_bit
    decrease_inventory_count(bitboard, figure_index)
    new_bitboard[IS_BLACK_TURN] = int(not new_bitboard[IS_BLACK_TURN])

    update_masks(new_bitboard)
    return new_bitboard