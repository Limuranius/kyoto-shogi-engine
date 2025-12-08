import numpy as np

import speed_analyzer
from .Bitboard import *
import bitarray5x5

BitboardBatch = np.ndarray  # Array (bitboard_size, N), where N - number of bitboards, bitboard_size - size of bitboard


def bitboard_to_batch(bitboard: Bitboard):
    return bitboard[None].T


@speed_analyzer.add_to_watchlist
def bits_positions(numbers: np.ndarray, return_shifts=True):
    """
    Takes n-dimensional array of integers as input.
    Returns (n+1)-dimensional array of positions of bits in these integers
    Last dimension will store number of shifts of each bit (if return_shifts is set to True),
    or separate bits (if return_shifts is set to False)
    For example, 19 (0b10011) will be represented as [0, 1, 4]
    If number of bits is different in numbers, then extra bits will be padded at the end
    """
    numbers = numbers.copy()
    max_bit_count = np.bitwise_count(numbers).max()
    # dtype = np.uint8 if return_shifts else numbers.dtype
    dtype = numbers.dtype
    result = np.zeros(shape=(*numbers.shape, max_bit_count), dtype=dtype)
    for i in range(max_bit_count):
        lsb = numbers & -numbers
        if return_shifts:
            result[..., i] = np.bitwise_count(lsb - 1)
        else:
            result[..., i] = lsb
        numbers ^= lsb
    return result


@speed_analyzer.add_to_watchlist
def unrag_by_row(
        arr: np.ndarray,
        mask: np.ndarray,
) -> list[np.ndarray]:
    """
    Takes 2+ dimensional ragged array and 2d mask
    Applies mask on array and returns only masked values stored in list of rows of masked values
    """
    split_points = mask.sum(axis=1).cumsum()[:-1]  # Find where to split - count non-zero per row
    filtered = arr[mask]  # Apply mask to get filtered values
    return np.split(filtered, split_points)  # Split into list of arrays


@speed_analyzer.add_to_watchlist
def get_bitboards_moves(bitboards: BitboardBatch) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns all possible moves of bitboards. For each board there are two arrays:
        first array - all moves on the board. Move is stored as three integers: old position, new position, and figure type.
            Array of shape (M, 3), where M - number of moves
        index 1 - all drops. Drop is stored as two integers: position of drop and figure type.
            Array of shape (D, 2), where D - number of drops

    All bitboards have to have same turn color (white or black)
    """

    n_boards = bitboards.shape[1]

    figures = BLACK_FIGURE_INDICES
    friend_mask = IS_BLACK
    figures_to_drop = BLACK_DROP_FIGURE_INDICES
    if not bitboards[IS_BLACK_TURN, 0]:
        figures = WHITE_FIGURE_INDICES
        friend_mask = IS_WHITE
        figures_to_drop = WHITE_DROP_FIGURE_INDICES

    all_moves_by_board = [list() for _ in range(n_boards)]
    all_drops_by_board = [list() for _ in range(n_boards)]

    # iterating all moves
    for figure_index in figures:

        # get positions of figures in all boards
        shifts = bits_positions(bitboards[figure_index])  # shape = (n_boards, max_figure_count)
        if shifts.size == 0:  # there are no such figure through all boards
            continue
        # get attack masks for all positions
        # value 32 in :shifts gets attack mask of 0
        # same shape as :shifts
        attack_mask = get_figure_attack_mask_from_shift(
            figure_index,
            np.broadcast_to(bitboards[IS_OCCUPIED][:, None], shape=shifts.shape),
            shifts
        )
        attack_mask &= ~bitboards[friend_mask, :, None]  # removing friendly-fire attacks

        # shape = (n_boards, max_figure_count, max_attacks_count)
        destination_bits = bits_positions(attack_mask, return_shifts=False)  # ending cell of move

        # shape = (n_boards, max_figure_count)
        origin_bits = 1 << shifts  # starting cell of move

        # array of moves of shape (n_boards, max_figure_count, max_attacks_count, 3)
        moves = np.stack([
            np.broadcast_to(origin_bits[:, :, None], destination_bits.shape),  # origin
            destination_bits,  # destination
            np.broadcast_to(np.uint32(figure_index), destination_bits.shape)  # figure index
        ], axis=-1)

        # flattening extra dimensions
        # now shape is (n_boards, n_moves, 3)
        moves = moves.reshape((n_boards, -1, 3))

        valid_move_mask = moves[:, :, 1] != 0  # If destination is zero then this move is padding

        # removing all paddings
        # now moves are stored as list of (n_moves, 3) arrays - set of moves for each board
        moves_for_each_board = unrag_by_row(moves, valid_move_mask)

        for i in range(n_boards):
            all_moves_by_board[i].append(moves_for_each_board[i])

    for i in range(n_boards):
        if len(all_moves_by_board[i]) == 0:  # No moves on board
            all_moves_by_board[i] = np.zeros((0, 2), dtype=np.uint32)
        else:
            all_moves_by_board[i] = np.concat(all_moves_by_board[i], axis=0)

    drop_destinations = bits_positions(bitboards[IS_EMPTY], return_shifts=False)  # shape = (n_boards, max_empty_count)
    for figure_index in figures_to_drop:
        hand_count = get_inventory_count(bitboards, figure_index)
        has_hand = hand_count > 0

        # shape = (n_boards, max_empty_count, 2)
        drops = np.stack([
            drop_destinations,  # drop destination
            np.broadcast_to(np.uint32(figure_index), drop_destinations.shape)  # figure index
        ], axis=-1)

        valid_drop_mask = drops[:, :, 0] != 0  # If destination is zero then this drop is padding

        # removing all paddings
        # now moves are stored as list of (n_moves, 3) arrays - set of moves for each board
        drops_for_each_board = unrag_by_row(drops, valid_drop_mask)

        for i in range(n_boards):
            if has_hand[i]:
                all_drops_by_board[i].append(drops_for_each_board[i])

    for i in range(n_boards):
        if len(all_drops_by_board[i]) == 0:  # No drops on board
            all_drops_by_board[i] = np.zeros((0, 2), dtype=np.uint32)
        else:
            all_drops_by_board[i] = np.concat(all_drops_by_board[i], axis=0)

    return list(zip(all_moves_by_board, all_drops_by_board))


@speed_analyzer.add_to_watchlist
def update_batch_masks(bitboards: BitboardBatch) -> None:
    """Updates all masks of bitboards according to placement of pieces"""
    if bitboards.shape[1] == 0:  # empty batch
        return

    n_boards = bitboards.shape[1]
    bitboards[IS_BLACK] = np.bitwise_or.reduce(bitboards[TOKIN_BLACK: TOKIN_WHITE], axis=0)
    bitboards[IS_WHITE] = np.bitwise_or.reduce(bitboards[TOKIN_WHITE: ROOK_WHITE + 1], axis=0)
    bitboards[IS_OCCUPIED] = bitboards[IS_BLACK] | bitboards[IS_WHITE]
    bitboards[IS_EMPTY] = ALL_BITS ^ bitboards[IS_OCCUPIED]

    black_attacks_mask = 0
    white_attacks_mask = 0
    for figure_index in FIGURE_INDICES:
        shifts = bits_positions(bitboards[figure_index])  # shape = (n_boards, max_figure_count)
        if shifts.size == 0:  # there are no such figure through all boards
            continue
        attack_mask = get_figure_attack_mask_from_shift(
            figure_index,
            np.broadcast_to(bitboards[IS_OCCUPIED][:, None], shape=shifts.shape),
            shifts
        )  # same shape as shifts
        full_attack_mask = np.bitwise_or.reduce(attack_mask, axis=1)

        if IS_FIGURE_BLACK[figure_index]:
            black_attacks_mask |= full_attack_mask  # add attack to total mask
        else:
            white_attacks_mask |= full_attack_mask

        # Setting all selected indices to figure type
        bitboards[FIGURE_AT_FLAT[shifts], np.arange(n_boards)[:, None]] = figure_index

    bitboards[ATTACKS_FF_BLACK] = black_attacks_mask
    bitboards[ATTACKS_FF_WHITE] = white_attacks_mask
    bitboards[ATTACKS_BLACK] = black_attacks_mask & ~bitboards[IS_BLACK]
    bitboards[ATTACKS_WHITE] = white_attacks_mask & ~bitboards[IS_WHITE]


@speed_analyzer.add_to_watchlist
def make_moves_fast(
        bitboard: Bitboard | BitboardBatch,
        prev_pos_bit: np.ndarray,
        new_pos_bit: np.ndarray,
        figure_index: np.ndarray,
) -> BitboardBatch:
    """
    prev_pos_bit - bit on the board, where piece was before the move
    new_pos_bit - bit on the board, where piece was after the move

    Assumptions for input (not checked in this function):
        Previous figure position is not empty
        Previous figure color matches current turn

    Does NOT perform update on batch of boards. Update needs to be performed somewhere else
    """
    n_moves = len(prev_pos_bit)
    if bitboard.ndim == 1:  # is bitboard
        batch = np.repeat(
            bitboard[:, None],
            repeats=n_moves,
            axis=1
        )
    else:  # is batch
        batch = bitboard
    idx = np.arange(n_moves)

    batch[FLIP_FIGURE[figure_index], idx] |= new_pos_bit
    batch[figure_index, idx] ^= prev_pos_bit

    # Increasing inventory count for moves that were takes
    take_mask = (batch[IS_OCCUPIED] & new_pos_bit).astype(bool)
    new_shifts = bitarray5x5.bit_to_shift(new_pos_bit)
    taken_figure_index = batch[FIGURE_AT_FLAT[new_shifts], idx]
    taken_figure_index[~take_mask] = TRASH  # will be adding +1 to trash for all non-take moves
    increase_inventory_count(batch, FLIP_COLOR[taken_figure_index])
    batch[taken_figure_index, idx] ^= new_pos_bit  # remove taken piece

    # Flipping turn
    batch[IS_BLACK_TURN] = ~batch[IS_BLACK_TURN].astype(bool)

    return batch


@speed_analyzer.add_to_watchlist
def make_drops_fast(
        bitboard: Bitboard | BitboardBatch,
        drop_pos_bit: np.ndarray,
        figure_index: np.ndarray,
) -> BitboardBatch:
    n_moves = len(drop_pos_bit)
    if bitboard.ndim == 1:  # is bitboard
        batch = np.repeat(
            bitboard[:, None],
            repeats=n_moves,
            axis=1
        )
    else:  # is batch
        batch = bitboard
    idx = np.arange(n_moves)

    batch[figure_index, idx] |= drop_pos_bit
    decrease_inventory_count(batch, figure_index)
    batch[IS_BLACK_TURN] = ~batch[IS_BLACK_TURN].astype(bool)

    return batch


@speed_analyzer.add_to_watchlist
def make_batch_moves_and_drops(
        bitboards: BitboardBatch,
        moves_and_drops: list[tuple[np.ndarray, np.ndarray]],
        concat: bool,
) -> BitboardBatch | list[BitboardBatch]:
    if concat:
        # repeat boards according to number of moves for each board
        n_moves = [len(moves) for moves, drops in moves_and_drops]
        n_drops = [len(drops) for moves, drops in moves_and_drops]
        bitboards_move = np.repeat(bitboards, repeats=n_moves, axis=1)
        bitboards_drop = np.repeat(bitboards, repeats=n_drops, axis=1)

        moves = np.concat([i[0] for i in moves_and_drops])
        drops = np.concat([i[1] for i in moves_and_drops])

        bitboards_move = make_moves_fast(bitboards_move, *moves.T)
        bitboards_drop = make_drops_fast(bitboards_drop, *drops.T)

        new_bitboards = np.concat([bitboards_move, bitboards_drop], axis=1)
        update_batch_masks(new_bitboards)
        return new_bitboards
    else:
        new_bitboards = []
        for i, (moves, drops) in enumerate(moves_and_drops):
            bitboard = bitboards[:, i]
            new_bitboards.append(make_moves_fast(bitboard, *moves.T))
            new_bitboards.append(make_drops_fast(bitboard, *drops.T))

        result = []
        # concatenating moves and drops for each board separately
        for i in range(0, len(new_bitboards), 2):
            boards = np.concat(new_bitboards[i: i + 2], axis=1)
            update_batch_masks(boards)
            result.append(boards)
        return result
