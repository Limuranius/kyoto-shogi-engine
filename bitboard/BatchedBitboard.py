import numpy as np
from .Bitboard import *

BatchedBitboard = np.ndarray  # Array (bitboard_size, N), where N - number of bitboards, bitboard_size - size of bitboard


def bits_positions(numbers: np.ndarray, return_shifts=True):
    """
    Takes n-dimensional array of integers as input.
    Returns (n+1)-dimensional array of positions of bits in these integers
    Last dimension will store number of shifts of each bit (if return_shifts is set to True),
    or separate bits (if return_shifts is set to False)
    For example, 19 (0b10011) will be represented as [0, 1, 4]
    If number of bits is different in numbers, then extra bits will be padded at the end
    """
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


def get_bitboards_moves(bitboards: BatchedBitboard) -> list[tuple[np.ndarray, np.ndarray]]:
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
        attack_mask = get_figure_attack_mask_from_shift(figure_index, bitboards[IS_OCCUPIED], shifts)
        attack_mask &= ~bitboards[friend_mask, None]  # removing friendly-fire attacks

        # shape = (n_boards, max_figure_count, max_attacks_count)
        destination_bits = bits_positions(attack_mask, return_shifts=False)  # ending cell of move

        # shape = (n_boards, max_figure_count)
        origin_bits = 1 << shifts  # starting cell of move

        # array of moves of shape (n_boards, max_figure_count, max_attacks_count, 3)
        moves = np.stack([
            np.broadcast_to(origin_bits[:, :, None], destination_bits.shape),  # origin
            destination_bits,  # destination
            np.broadcast_to(figure_index, destination_bits.shape)  # figure index
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
        all_moves_by_board[i] = np.concat(all_moves_by_board[i], axis=0)


    drop_destinations = bits_positions(bitboards[IS_EMPTY], return_shifts=False)  # shape = (n_boards, max_empty_count)
    for figure_index in figures_to_drop:
        hand_count = get_inventory_count(bitboards, figure_index)
        has_hand = hand_count > 0

        # shape = (n_boards, max_empty_count, 2)
        drops = np.stack([
            drop_destinations,  # drop destination
            np.broadcast_to(figure_index, drop_destinations.shape)  # figure index
        ], axis=-1)

        valid_drop_mask = drops[:, :, 0] != 0  # If destination is zero then this drop is padding

        # removing all paddings
        # now moves are stored as list of (n_moves, 3) arrays - set of moves for each board
        drops_for_each_board = unrag_by_row(drops, valid_drop_mask)

        for i in range(n_boards):
            if has_hand[i]:
                all_drops_by_board[i].append(drops_for_each_board[i])

    for i in range(n_boards):
        all_drops_by_board[i] = np.concat(all_drops_by_board[i], axis=0)

    return list(zip(all_moves_by_board, all_drops_by_board))


