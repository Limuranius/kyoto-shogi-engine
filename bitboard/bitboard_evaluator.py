from .Bitboard import *
from .BitboardBatch import BitboardBatch
from .bitarray5x5 import bit_to_shift

INVENTORY_COST = np.zeros(18, dtype=float)
INVENTORY_COST[TOKIN_BLACK] = 1.0
INVENTORY_COST[LANCE_BLACK] = 1.0
INVENTORY_COST[SILVER_BLACK] = 1.0
INVENTORY_COST[BISHOP_BLACK] = 1.0
INVENTORY_COST[KING_BLACK] = 1000000.0
INVENTORY_COST[GOLD_BLACK] = 1.0
INVENTORY_COST[KNIGHT_BLACK] = 1.0
INVENTORY_COST[PAWN_BLACK] = 1.0
INVENTORY_COST[ROOK_BLACK] = 1.0
INVENTORY_COST[TOKIN_WHITE] = 1.0
INVENTORY_COST[LANCE_WHITE] = 1.0
INVENTORY_COST[SILVER_WHITE] = 1.0
INVENTORY_COST[BISHOP_WHITE] = 1.0
INVENTORY_COST[KING_WHITE] = 1000000.0
INVENTORY_COST[GOLD_WHITE] = 1.0
INVENTORY_COST[KNIGHT_WHITE] = 1.0
INVENTORY_COST[PAWN_WHITE] = 1.0
INVENTORY_COST[ROOK_WHITE] = 1.0

BOARD_COST = np.zeros(18, dtype=float)
BOARD_COST[TOKIN_BLACK] = 1.0
BOARD_COST[LANCE_BLACK] = 1.0
BOARD_COST[SILVER_BLACK] = 1.0
BOARD_COST[BISHOP_BLACK] = 1.0
BOARD_COST[KING_BLACK] = 0.0
BOARD_COST[GOLD_BLACK] = 1.0
BOARD_COST[KNIGHT_BLACK] = 1.0
BOARD_COST[PAWN_BLACK] = 1.0
BOARD_COST[ROOK_BLACK] = 1.0
BOARD_COST[TOKIN_WHITE] = 1.0
BOARD_COST[LANCE_WHITE] = 1.0
BOARD_COST[SILVER_WHITE] = 1.0
BOARD_COST[BISHOP_WHITE] = 1.0
BOARD_COST[KING_WHITE] = 0.0
BOARD_COST[GOLD_WHITE] = 1.0
BOARD_COST[KNIGHT_WHITE] = 1.0
BOARD_COST[PAWN_WHITE] = 1.0
BOARD_COST[ROOK_WHITE] = 1.0

ATTACK_COUNT_WEIGHT = 1.0
DEFENCE_COUNT_WEIGHT = 1.0
KING_DEFENCE_COUNT_WEIGHT = 3.0
KING_ATTACK_COUNT_WEIGHT = 3.0

BLACK = 1
WHITE = 0


class BitboardEvaluator:
    def evaluate_board(self, bitboard: Bitboard | BitboardBatch) -> float | np.ndarray:
        methods = [
            (self._inventory_cost, 1.0),
            (self._board_cost, 1.0),
            (self._attack_count, ATTACK_COUNT_WEIGHT),
            (self._defence_count, DEFENCE_COUNT_WEIGHT),
            (self._king_defence_count, KING_DEFENCE_COUNT_WEIGHT),
            (self._king_attack_count, KING_ATTACK_COUNT_WEIGHT),
        ]
        total_score = 0
        for method, weight in methods:
            total_score += method(bitboard, BLACK) * weight
            total_score -= method(bitboard, WHITE) * weight

        return total_score

    @speed_analyzer.add_to_watchlist
    def _inventory_cost(self, bitboard: Bitboard | BitboardBatch, side: int) -> float | np.ndarray:
        """Cost of pieces in inventory"""
        score = 0.0
        if side == BLACK:
            figures = BLACK_DROP_FIGURE_INDICES
        else:
            figures = WHITE_DROP_FIGURE_INDICES

        for fig in figures:
            score += get_inventory_count(bitboard, fig) * INVENTORY_COST[fig]
        return score

    @speed_analyzer.add_to_watchlist
    def _board_cost(self, bitboard: Bitboard | BitboardBatch, side: int) -> float | np.ndarray:
        """Cost of pieces on board"""
        score = 0.0

        if side == BLACK:
            figures = BLACK_FIGURE_INDICES
        else:
            figures = WHITE_FIGURE_INDICES

        for fig in figures:
            count = np.bitwise_count(bitboard[fig])
            # count = int(bitboard[fig]).bit_count()
            score += count * BOARD_COST[fig]
        return score

    @speed_analyzer.add_to_watchlist
    def _attack_count(self, bitboard: Bitboard | BitboardBatch, side: int) -> float | np.ndarray:
        """Count of attacks on cells. If two pieces attack same cell then this counts as 2"""
        if side == BLACK:
            return np.bitwise_count(bitboard[ATTACKS_BLACK])
        else:
            return np.bitwise_count(bitboard[ATTACKS_WHITE])

    @speed_analyzer.add_to_watchlist
    def _defence_count(self, bitboard: Bitboard | BitboardBatch, side: int) -> float | np.ndarray:
        """Count of defences of friendly pieces. If two pieces defend same cell then this counts as 2"""
        if side == BLACK:
            return np.bitwise_count(bitboard[ATTACKS_BLACK] ^ bitboard[ATTACKS_FF_BLACK])
        else:
            return np.bitwise_count(bitboard[ATTACKS_WHITE] ^ bitboard[ATTACKS_FF_WHITE])

    @speed_analyzer.add_to_watchlist
    def _king_defence_count(self, bitboard: Bitboard | BitboardBatch, side: int) -> float | np.ndarray:
        """Count of defenced cells around king"""
        if side == BLACK:
            king = KING_BLACK
            attacks = ATTACKS_BLACK
            attacks_ff = ATTACKS_FF_BLACK
        else:
            king = KING_WHITE
            attacks = ATTACKS_WHITE
            attacks_ff = ATTACKS_FF_WHITE
        king_on_board_mask = bitboard[king] != 0
        king_shift = bit_to_shift(bitboard[king])
        king_surroundings = get_figure_attack_mask_from_shift(
            figure_index=king,
            figure_mask=bitboard[IS_OCCUPIED],
            shift=king_shift,
        )
        defence_mask = bitboard[attacks] ^ bitboard[attacks_ff]
        return king_on_board_mask * np.bitwise_count(king_surroundings & defence_mask)

    @speed_analyzer.add_to_watchlist
    def _king_attack_count(self, bitboard: Bitboard | BitboardBatch, side: int) -> float | np.ndarray:
        """Count of attacked cells around king"""
        if side == BLACK:
            king = KING_BLACK
            opposite_attacks = ATTACKS_WHITE
        else:
            king = KING_WHITE
            opposite_attacks = ATTACKS_BLACK

        king_on_board_mask = bitboard[king] != 0
        king_shift = bit_to_shift(bitboard[king])
        king_surroundings = get_figure_attack_mask_from_shift(
            figure_index=king,
            figure_mask=bitboard[IS_OCCUPIED],
            shift=king_shift,
        )
        return -(king_on_board_mask * np.bitwise_count(king_surroundings & bitboard[opposite_attacks])).astype(int)
