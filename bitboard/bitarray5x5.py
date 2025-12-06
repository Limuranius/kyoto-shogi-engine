import numpy as np

import speed_analyzer

VectorizableInt = int | np.ndarray


class Bitarray5x5:
    @staticmethod
    def shift_right(n: VectorizableInt):
        return (n >> 1) & 0b01111_01111_01111_01111_01111

    @staticmethod
    def shift_left(n: VectorizableInt):
        return (n << 1) & 0b11110_11110_11110_11110_11110

    @staticmethod
    def shift_up(n: VectorizableInt):
        return n << 5 & 0b11111_11111_11111_11111_11111

    @staticmethod
    def shift_down(n: VectorizableInt):
        return n >> 5

    @staticmethod
    def repeat_shift(n: VectorizableInt, shift_func, count: int):
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
    def pretty_print(n: int):
        empty = "░░"
        occupied = "██"
        bin_str = bin(n)[2:]
        bin_str = bin_str.zfill(25)  # fill leading zeros if len < 25
        for i in range(5):
            s = bin_str[i * 5: (i + 1) * 5]
            s = s.replace("1", occupied)
            s = s.replace("0", empty)
            print(s)

    @staticmethod
    def row_to_column(row: VectorizableInt, col_j: VectorizableInt) -> VectorizableInt:
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
    def get_column(board: VectorizableInt, col_j: VectorizableInt) -> VectorizableInt:
        row = 0
        row |= board >> (5 * 4 - col_j + 0) & 16
        row |= board >> (5 * 3 - col_j + 1) & 8
        row |= board >> (5 * 2 - col_j + 2) & 4
        row |= board >> (5 * 1 - col_j + 3) & 2
        row |= board >> (5 * 0 - col_j + 4) & 1
        return row & ROWS[-1]

    @staticmethod
    def get_row(board: VectorizableInt, row_i: VectorizableInt) -> VectorizableInt:
        return (board >> (5 * (4 - row_i))) & ROWS[-1]

    @staticmethod
    def main_diagonal_to_row(board: VectorizableInt, shifts: VectorizableInt) -> VectorizableInt:
        # Shifts of 5 cells in diagonal. 31 if cell out of bounds
        diag_shifts = MAIN_DIAGONAL_SHIFTS_FLAT[shifts]
        s0 = diag_shifts[..., 0]
        s1 = diag_shifts[..., 1]
        s2 = diag_shifts[..., 2]
        s3 = diag_shifts[..., 3]
        s4 = diag_shifts[..., 4]

        row = 0
        row |= (board & (1 << s0)) >> (s0 - 0)
        row |= (board & (1 << s1)) >> (s1 - 1)
        row |= (board & (1 << s2)) >> (s2 - 2)
        row |= (board & (1 << s3)) >> (s3 - 3)
        row |= (board & (1 << s4)) >> (s4 - 4)
        return row

    @staticmethod
    def secondary_diagonal_to_row(board: VectorizableInt, shifts: VectorizableInt) -> VectorizableInt:
        # Shifts of 5 cells in diagonal. 31 if cell out of bounds
        diag_shifts = SECONDARY_DIAGONAL_SHIFTS_FLAT[shifts]
        s0 = diag_shifts[..., 0]
        s1 = diag_shifts[..., 1]
        s2 = diag_shifts[..., 2]
        s3 = diag_shifts[..., 3]
        s4 = diag_shifts[..., 4]

        row = 0
        row |= (board & (1 << s0)) >> (s0 - 0)
        row |= (board & (1 << s1)) >> (s1 - 1)
        row |= (board & (1 << s2)) >> (s2 - 2)
        row |= (board & (1 << s3)) >> (s3 - 3)
        row |= (board & (1 << s4)) >> (s4 - 4)
        return row

    @staticmethod
    def shift_to_main_diagonal(board: VectorizableInt, shift: VectorizableInt) -> VectorizableInt:
        s0, s1, s2, s3, s4 = MAIN_DIAGONAL_SHIFTS_FLAT[shift]
        row = 0
        row |= (board & (1 << s0)) >> (s0 - 0)
        row |= (board & (1 << s1)) >> (s1 - 1)
        row |= (board & (1 << s2)) >> (s2 - 2)
        row |= (board & (1 << s3)) >> (s3 - 3)
        row |= (board & (1 << s4)) >> (s4 - 4)
        return row


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

PositionBit = int  # bit stored in integer, for example: 0b00001000
ShiftCount = int  # number of shifts of bit from zero, for example: shift count 3 for number 0b00001000
BitCoord2D = tuple[int, int]  # (i, j) coordinate of bit in 5x5 grid
BitMask = int  # multiple bits in one integer, for example 0b001010011


def get_bit_coord(n: PositionBit) -> BitCoord2D:
    # pos = np.bitwise_count(n - 1)
    pos = (n - 1).bit_count()
    i = 4 - pos // 5
    j = 4 - pos % 5
    return i, j


@speed_analyzer.add_to_watchlist
def get_bits_coords(n: BitMask) -> list[BitCoord2D]:
    """Returns list of (i, j) coordinates of each bit in 5x5 grid"""
    coords = []
    while n:
        lsb = n & -n  # least significant bit
        # pos = np.bitwise_count(lsb - 1).astype(np.uint32)
        pos = (lsb - 1).bit_count()
        i = 4 - pos // 5
        j = 4 - pos % 5
        coords.append((i, j))
        n ^= lsb  # remove lowest bit
    return coords


def position_mask_from_coordinates(coords: list[BitCoord2D]) -> BitMask:
    result = 0
    for i, j in coords:
        n_shifts = (4 - i) * 5 + (4 - j)
        result |= 1 << n_shifts
    return result


def coord_to_bit(i: VectorizableInt, j: VectorizableInt) -> PositionBit:
    return 1 << coord_to_shift(i, j)


def coord_to_shift(i: VectorizableInt, j: VectorizableInt) -> ShiftCount:
    return (4 - i) * 5 + (4 - j)


def shift_to_col(shift: VectorizableInt) -> VectorizableInt:
    return 4 - (shift % 5)


def shift_to_row(shift: VectorizableInt) -> VectorizableInt:
    return 4 - (shift // 5)

def bit_to_shift(bit: VectorizableInt) -> VectorizableInt:
    return np.bitwise_count(bit - 1)

"""
Similar to get_main_diagonal_coords(i, j) and get_secondary_diagonal_coords(i, j)
but returns array of 5 shifts of each cell in diagonal
Value 32 means cell is out of bounds

For example:
MAIN_DIAGONAL_SHIFTS_FLAT[9] - shifts of main diagonal cells that passes through cell at shift 9 (i=3, j=0)
MAIN_DIAGONAL_SHIFTS_FLAT[9] == [3, 9, 32, 32, 32]
"""
MAIN_DIAGONAL_SHIFTS_FLAT = np.full(shape=(33, 5), dtype=np.uint32, fill_value=32)
SECONDARY_DIAGONAL_SHIFTS_FLAT = np.full(shape=(33, 5), dtype=np.uint32, fill_value=32)
for i in range(5):
    for j in range(5):
        for ii, (diag_i, diag_j) in enumerate(reversed(get_main_diagonal_coords(i, j))):
            MAIN_DIAGONAL_SHIFTS_FLAT[coord_to_shift(i, j), ii] = coord_to_shift(diag_i, diag_j)
        for ii, (diag_i, diag_j) in enumerate(get_secondary_diagonal_coords(i, j)):
            SECONDARY_DIAGONAL_SHIFTS_FLAT[coord_to_shift(i, j), ii] = coord_to_shift(diag_i, diag_j)

# shift, pos_bit, coord
