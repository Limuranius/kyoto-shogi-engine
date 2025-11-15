from utils import *


def rook_moves(i: int, j: int) -> int:
    return (ROWS[i] | COLS[j]) & ~CELLS[i][j]

def bishop_moves(i: int, j: int) -> int:
    main_diag = MAIN_DIAGONALS[j - i + 4]
    sec_diag = SECONDARY_DIAGONALS[i + j]
    return (main_diag | sec_diag) & ~CELLS[i][j]






Bitarray5x5.print(bishop_moves(2, 1))



class Bitboard:
    # Положение фигур

    # Атаки фигур (без учёта положений)

    pass