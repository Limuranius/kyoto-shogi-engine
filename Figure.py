from __future__ import annotations
import enum
from typing import Literal
import numpy as np


class Side(enum.Enum):
    BLACK = "U"
    WHITE = "D"

    def opposite(self):
        if self == Side.BLACK:
            return Side.WHITE
        return Side.BLACK


class FigureType(enum.Enum):
    TOKIN_LANCE = enum.auto()
    SILVER_BISHOP = enum.auto()
    KING = enum.auto()
    GOLD_KNIGHT = enum.auto()
    PAWN_ROOK = enum.auto()

    def to_jp(self) -> tuple[str, str]:
        translate_table = {
            self.TOKIN_LANCE: ("と", "香"),
            self.SILVER_BISHOP: ("銀", "角"),
            self.KING: ("玉", "玉"),
            self.GOLD_KNIGHT: ("金", "桂"),
            self.PAWN_ROOK: ("歩", "飛"),
        }
        return translate_table[self]


class Figure:
    side: Side
    type: FigureType
    state: Literal[0, 1]

    def __init__(self, type: FigureType, side: Side, state: Literal[0, 1] = 0):
        self.side = side
        self.type = type
        self.state = state

    def get_moves(self):
        return figure_moves[(self.type, self.state, self.side)]

    def flipped(self) -> Figure:
        return Figure(self.type, self.side, int(self.state == 0))

    def __repr__(self):
        return f"{self.type}[{self.state}] {self.side}"

    def to_jp(self) -> str:
        return self.type.to_jp()[self.state]

    @classmethod
    def from_jp(self, ch: str, side: Side = Side.BLACK) -> Figure:
        jp_to_type = {
            "と": (FigureType.TOKIN_LANCE, 0),
            "香": (FigureType.TOKIN_LANCE, 1),
            "銀": (FigureType.SILVER_BISHOP, 0),
            "角": (FigureType.SILVER_BISHOP, 1),
            "玉": (FigureType.KING, 0),
            "金": (FigureType.GOLD_KNIGHT, 0),
            "桂": (FigureType.GOLD_KNIGHT, 1),
            "歩": (FigureType.PAWN_ROOK, 0),
            "飛": (FigureType.PAWN_ROOK, 1),
        }
        figure_type, state = jp_to_type[ch]
        return Figure(figure_type, side=side, state=state)

    def __eq__(self, other):
        if other is None:
            return False
        type_eq = self.type == other.type
        side_eq = self.side == other.side
        state_eq = self.state == other.state
        is_king = self.type == FigureType.KING
        return (type_eq and side_eq and state_eq) or (type_eq and is_king and side_eq)


# Ходы
king_moves = {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)}
rook_moves = {
    *[(0, dx) for dx in range(-4, 5) if dx != 0],
    *[(dy, 0) for dy in range(-4, 5) if dy != 0]
}
bishop_moves = {
    *[(d, d) for d in range(-4, 5) if d != 0],
    *[(d, -d) for d in range(-4, 5) if d != 0],
}
gold_moves = {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)}
figure_moves = {  # (FigureType, state)
    (FigureType.TOKIN_LANCE, 0): gold_moves,
    (FigureType.TOKIN_LANCE, 1): {(-dy, 0) for dy in range(1, 5)},
    (FigureType.SILVER_BISHOP, 0): {(-1, -1), (-1, 0), (1, -1), (-1, 1), (1, 1)},
    (FigureType.SILVER_BISHOP, 1): bishop_moves,
    (FigureType.KING, 0): king_moves,
    (FigureType.KING, 1): king_moves,
    (FigureType.GOLD_KNIGHT, 0): gold_moves,
    (FigureType.GOLD_KNIGHT, 1): {(-2, -1), (-2, 1)},
    (FigureType.PAWN_ROOK, 0): {(-1, 0)},
    (FigureType.PAWN_ROOK, 1): rook_moves,
}
new_figure_moves = dict()  # (FigureType, state, side)
for figure_type, state in figure_moves:
    moves = figure_moves[(figure_type, state)]
    inv_moves = {(-dy, dx) for dy, dx in moves}
    new_figure_moves[(figure_type, state, Side.BLACK)] = moves
    new_figure_moves[(figure_type, state, Side.WHITE)] = inv_moves
figure_moves = new_figure_moves
figure_moves = {fig: np.array(list(figure_moves[fig])) for fig in figure_moves}
