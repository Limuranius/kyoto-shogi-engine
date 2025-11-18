import numpy as np

from .utils import *

N_PUBLIC_FIELDS = 28  # number of fields in bitboard that we can access with named variable (public fields)
N_FIELDS = N_PUBLIC_FIELDS + 25  # +25 integers for indices of figures in each cell
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
) = range(N_PUBLIC_FIELDS)

# for each (i, j) coordinate stores index of integer inside bitboard that stores index of figure in (i, j) cell
FIGURE_AT = np.arange(N_PUBLIC_FIELDS, N_PUBLIC_FIELDS + 25, dtype=np.uint32).reshape((5, 5))

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
BLACK_FIGURE_INDICES = list(range(TOKIN_BLACK, TOKIN_WHITE))
WHITE_FIGURE_INDICES = list(range(TOKIN_WHITE, ROOK_WHITE + 1))
BLACK_DROP_FIGURE_INDICES = [TOKIN_BLACK, SILVER_BLACK, KING_BLACK, GOLD_BLACK, PAWN_BLACK]
WHITE_DROP_FIGURE_INDICES = [TOKIN_WHITE, SILVER_WHITE, KING_WHITE, GOLD_WHITE, PAWN_WHITE]


def get_empty_bitboard() -> Bitboard:
    return np.zeros(N_FIELDS, dtype=np.uint32)


def get_bit_coord(n: PositionBit) -> tuple[int, int]:
    pos = np.bitwise_count(n - 1)
    i = 4 - pos // 5
    j = 4 - pos % 5
    return i, j


def get_bits_coords(n: int) -> list[tuple[int, int]]:
    """Returns list of (i, j) coordinates of each bit in 5x5 grid"""
    coords = []
    while n:
        lsb = n & -n  # least significant bit
        pos = np.bitwise_count(lsb - 1).astype(np.uint32)
        i = 4 - pos // 5
        j = 4 - pos % 5
        coords.append((i, j))
        n ^= lsb  # remove lowest bit
    return coords


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

    bitboard[IS_BLACK] = np.bitwise_or.reduce(bitboard[TOKIN_BLACK: TOKIN_WHITE])
    bitboard[IS_WHITE] = np.bitwise_or.reduce(bitboard[TOKIN_WHITE: ROOK_WHITE + 1])
    bitboard[IS_OCCUPIED] = bitboard[IS_BLACK] | bitboard[IS_WHITE]
    bitboard[IS_EMPTY] = 0 ^ bitboard[IS_OCCUPIED]

    for figure_index in FIGURE_INDICES:
        position_mask = bitboard[figure_index]
        for i, j in get_bits_coords(position_mask):  # iterating through figures positions
            # Storing attacked cells
            attack_mask = get_figure_attack_mask(figure_index, bitboard[IS_OCCUPIED], i, j)
            if IS_FIGURE_BLACK[figure_index]:
                black_attacks_mask |= attack_mask  # add attack to total mask
            else:
                white_attacks_mask |= attack_mask

            # Storing figure type at (i, j) cell
            bitboard[FIGURE_AT[i, j]] = figure_index

    bitboard[ATTACKS_FF_BLACK] = black_attacks_mask
    bitboard[ATTACKS_FF_WHITE] = white_attacks_mask
    bitboard[ATTACKS_BLACK] = black_attacks_mask & ~bitboard[IS_BLACK]
    bitboard[ATTACKS_WHITE] = white_attacks_mask & ~bitboard[IS_WHITE]


def get_figure_index(
        bitboard: Bitboard,
        cell_bit: PositionBit
) -> BitboardIndex:
    # return (bitboard[0: ROOK_WHITE + 1] & cell_bit).argmax()
    i, j = get_bit_coord(cell_bit)
    return bitboard[FIGURE_AT[i, j]]


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


def position_mask_from_coordinates(coords: list[tuple[int, int]]):
    result = 0
    for i, j in coords:
        n_shifts = (4 - i) * 5 + (4 - j)
        result |= 1 << n_shifts
    return result


def position_bit_from_coordinates(i: int, j: int) -> int:
    n_shifts = (4 - i) * 5 + (4 - j)
    return 1 << n_shifts


def figure_at(bitboard: Bitboard, i: int, j: int) -> int:
    return bitboard[FIGURE_AT[i, j]]


def get_bitboard_moves(bitboard: Bitboard) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns all possible moves of bitboard in two arrays:
        index 0 - all moves on the board. Move is stored as two integers: old position and new position.
            Array of shape (M, 2), where M - number of moves
        index 1 - all drops. Drop is stored as two integers: position of drop and figure type.
            Array of shape (D, 2), where D - number of drops
    """
    figures = BLACK_FIGURE_INDICES
    friend_mask = IS_BLACK
    figures_to_drop = BLACK_DROP_FIGURE_INDICES
    if not bitboard[IS_BLACK_TURN]:
        figures = WHITE_FIGURE_INDICES
        friend_mask = IS_WHITE
        figures_to_drop = WHITE_DROP_FIGURE_INDICES

    figures_to_drop = [f for f in figures_to_drop if get_inventory_count(bitboard, f) > 0]

    moves = np.zeros((50, 2), dtype=np.uint32)
    drops = np.zeros((25, 2), dtype=np.uint32)
    move_i = 0
    drop_i = 0

    # iterating all moves
    for figure_index in figures:
        for i, j in get_bits_coords(bitboard[figure_index]):
            pos_bit = position_bit_from_coordinates(i, j)
            attack_mask = get_figure_attack_mask(figure_index, bitboard[IS_OCCUPIED], i, j)
            attack_mask &= ~bitboard[friend_mask]  # removing friendly-fire attacks
            while attack_mask:  # iterating all bits of attack mask
                new_pos = -attack_mask & attack_mask  # least significant bit
                moves[move_i, 0] = pos_bit
                moves[move_i, 1] = new_pos
                move_i += 1
                attack_mask ^= new_pos  # set bit to zero

    # iterating all drops
    drop_mask = bitboard[IS_EMPTY]
    print(bin(drop_mask))
    while drop_mask:  # iterating all bits of free cells
        drop_pos = -drop_mask & drop_mask  # least significant bit
        for figure_index in figures_to_drop:
            drops[drop_i, 0] = drop_pos
            drops[drop_i, 1] = figure_index
            drop_i += 1
        drop_mask ^= drop_pos  # set bit to zero

    return moves[:move_i], drops[:drop_i]
