from collections.abc import Callable

from Board import Board
from Figure import Side, Figure, FigureType


WEIGHTS = {
    "inventory_cost": {
        FigureType.GOLD_KNIGHT: 1.0,
        FigureType.PAWN_ROOK: 1.0,
        FigureType.TOKIN_LANCE: 1.0,
        FigureType.SILVER_BISHOP: 1.0,
        FigureType.KING: 1000000.0,
    },
    "board_cost": {
        FigureType.GOLD_KNIGHT: 1.0,
        FigureType.PAWN_ROOK: 1.0,
        FigureType.TOKIN_LANCE: 1.0,
        FigureType.SILVER_BISHOP: 1.0,
        FigureType.KING: 0.0,
    },
    "attack_count": 1.0,
    "defence_count": 1.0,
    "king_defence_count": 3.0,
    "king_attack_count": 3.0,
}


class Evaluator:
    cache: dict[Board, float]

    def __init__(self):
        self.cache = dict()

    def evaluate_board(self, board: Board) -> float:
        if board in self.cache:
            return self.cache[board]

        methods = [
            (self._inventory_cost, 1.0),
            (self._board_cost, 1.0),
            (self._attack_count, WEIGHTS["attack_count"]),
            (self._defence_count, WEIGHTS["defence_count"]),
            (self._king_defence_count, WEIGHTS["king_defence_count"]),
            (self._king_attack_count, WEIGHTS["king_attack_count"]),
        ]
        total_score = 0
        for method, weight in methods:
            total_score += method(board, Side.BLACK) * weight
            total_score -= method(board, Side.WHITE) * weight

        self.cache[board] = total_score
        return total_score

    def _callback_same_side(
            self,
            board: Board,
            side: Side,
            callback: Callable[[
                tuple[int, int],  # i, j
                Figure
            ], None]
    ) -> None:
        for i in range(5):
            for j in range(5):
                if not board.is_empty_cell(i, j) and board.cells[i][j].side == side:
                    callback((i, j), board.cells[i][j])

    def calculate_attack_count(self, board: Board, side: Side):
        counts = [[0 for _ in range(5)] for _ in range(5)]
        def callback(coord: tuple[int, int], figure: Figure):
            for i, j in board.get_cell_destinations(*coord, include_friendly_fire=False):
                counts[i][j] += 1
        self._callback_same_side(board, side, callback)
        return counts

    def calculate_defence_count(self, board: Board, side: Side):
        counts = [[0 for _ in range(5)] for _ in range(5)]

        def callback(coord: tuple[int, int], figure: Figure):
            for i, j in board.get_cell_destinations(*coord, include_friendly_fire=True):
                if board.cells[i][j] is not None and board.cells[i][j].side == figure.side:
                    counts[i][j] += 1

        self._callback_same_side(board, side, callback)
        return counts

    def calculate_defence_and_attack_count(self, board: Board, side: Side):
        counts = [[0 for _ in range(5)] for _ in range(5)]

        def callback(coord: tuple[int, int], figure: Figure):
            for i, j in board.get_cell_destinations(*coord, include_friendly_fire=True):
                counts[i][j] += 1

        self._callback_same_side(board, side, callback)
        return counts

    def _inventory_cost(self, board: Board, side: Side) -> float:
        """Cost of pieces in inventory"""
        score = 0
        inv = board.get_side_inventory(side)
        for figure_type in inv:
            score += inv[figure_type] * WEIGHTS["inventory_cost"][figure_type]
        return score

    def _board_cost(self, board: Board, side: Side) -> float:
        """Cost of pieces on board"""
        score = 0

        def callback(coord: tuple[int, int], figure: Figure):
            nonlocal score
            score += WEIGHTS["board_cost"][figure.type]

        self._callback_same_side(board, side, callback)
        return score

    def _attack_count(self, board: Board, side: Side) -> float:
        """Count of attacks on cells. If two pieces attack same cell then this counts as 2"""
        attack_counts = self.calculate_attack_count(board, side)
        score = 0
        for i in range(5):
            for j in range(5):
                score += attack_counts[i][j]
        return score

    def _defence_count(self, board: Board, side: Side) -> float:
        """Count of defences of friendly pieces. If two pieces defend same cell then this counts as 2"""
        defence_counts = self.calculate_defence_count(board, side)
        score = 0
        for i in range(5):
            for j in range(5):
                score += defence_counts[i][j]
        return score

    def _king_defence_count(self, board: Board, side: Side) -> float:
        """Count of defenced cells around king"""
        if FigureType.KING in board.get_side_inventory(side.opposite()):  # King was taken
            return 0.0
        defence_counts = self.calculate_defence_and_attack_count(board, side)
        score = 0
        king_i, king_j = board.get_king_position(side)
        for i in range(5):
            for j in range(5):
                dist = max(abs(king_i - i), abs(king_j - j))
                if dist == 1:
                    pass
                score += defence_counts[i][j]
        return score

    def _king_attack_count(self, board: Board, side: Side) -> float:
        """Count of attacked cells around king"""
        if FigureType.KING in board.get_side_inventory(side.opposite()):  # King was taken
            return 0.0
        attack_counts = self.calculate_attack_count(board, side.opposite())
        score = 0
        king_i, king_j = board.get_king_position(side)
        for i in range(5):
            for j in range(5):
                dist = max(abs(king_i - i), abs(king_j - j))
                if dist == 1:
                    pass
                score += attack_counts[i][j]
        return -score
