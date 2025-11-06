from __future__ import annotations

import collections
import copy

from Figure import Figure, Side, FigureType
from Move import Move

Inventory = dict[FigureType, int]
BoardCells = list[list[Figure | None]]


class Board:
    cells: BoardCells
    inventory_black: Inventory
    inventory_white: Inventory
    turn: Side  # Which side has the move

    def __init__(
            self,
            cells: BoardCells,
            inventory_black: Inventory,
            inventory_white: Inventory,
            turn: Side = Side.WHITE,
    ):
        self.cells = cells
        self.inventory_black = inventory_black
        self.inventory_white = inventory_white
        self.turn = turn

    @classmethod
    def default_board(cls):
        cells = [
            [
                Figure(FigureType.PAWN_ROOK, Side.WHITE),
                Figure(FigureType.GOLD_KNIGHT, Side.WHITE),
                Figure(FigureType.KING, Side.WHITE),
                Figure(FigureType.SILVER_BISHOP, Side.WHITE),
                Figure(FigureType.TOKIN_LANCE, Side.WHITE),
            ],
            [None] * 5,
            [None] * 5,
            [None] * 5,
            [
                Figure(FigureType.TOKIN_LANCE, Side.BLACK),
                Figure(FigureType.SILVER_BISHOP, Side.BLACK),
                Figure(FigureType.KING, Side.BLACK),
                Figure(FigureType.GOLD_KNIGHT, Side.BLACK),
                Figure(FigureType.PAWN_ROOK, Side.BLACK),
            ],
        ]

        return Board(
            cells=cells,
            inventory_black=collections.defaultdict(int),
            inventory_white=collections.defaultdict(int)
        )

    def get_inventory_lists(self) -> tuple[list[FigureType], list[FigureType]]:
        black = []
        white = []
        for figure in self.inventory_black:
            count = self.inventory_black[figure]
            black += [figure] * count
        for figure in self.inventory_white:
            count = self.inventory_white[figure]
            white += [figure] * count
        return black, white

    def make_move(self, move: Move) -> Board:
        """Makes move and returns new version of board"""
        new_board = self.copy()
        i, j = move.array_destination
        inv = self.get_side_inventory(self.turn)
        assert self.turn == move.figure.side  # is right side to move?
        if move.is_drop:
            new_board.cells[i][j] = move.figure
            assert inv[move.figure.type] > 0
            inv[move.figure.type] -= 1
            if inv[move.figure.type] == 0:
                del inv[move.figure.type]
        else:
            if not self.is_empty_cell(i, j):  # If take
                assert self.cells[i][j].side != self.turn  # Check not friendly fire
                inv[self.cells[i][j].type] += 1  # Add taken figure to inventory
            orig_i, orig_j = move.array_origin
            new_board.cells[orig_i][orig_j] = None
            new_board.cells[i][j] = move.figure.flipped()
        new_board.turn = self.turn.opposite()
        return new_board

    def get_cell_moves(self, i: int, j: int) -> list[Move]:
        """
        Returns all possible moves for cell (i, j)
        If cell is empty returns all drops on this cell
        If cell is enemy's piece returns empty list
        If cell is current player's piece returns all valid moves of this figure
        """
        moves = []
        if self.cells[i][j] is None:
            # Cell empty. Checking what figures we can drop here
            if self.turn == Side.BLACK:
                inv = self.inventory_black
            else:
                inv = self.inventory_white
            figures_to_drop = [fig for fig in inv if inv[fig] > 0]
            for figure_type in figures_to_drop:
                moves.append(Move(
                    array_destination=(i, j),
                    figure=Figure(type=figure_type, side=self.turn, state=0),
                    is_drop=True,
                ))
                moves.append(Move(
                    array_destination=(i, j),
                    figure=Figure(type=figure_type, side=self.turn, state=1),
                    is_drop=True,
                ))
        else:
            if self.cells[i][j].side != self.turn:  # Enemy's figure
                return []
            figure = self.cells[i][j]
            for dy, dx in figure.get_moves():
                new_i, new_j = i + dy, j + dx
                if 0 <= new_i < 5 and 0 <= new_j < 5:  # in bounds
                    not_friendly_fire = any([
                        self.is_empty_cell(new_i, new_j),
                        self.cells[new_i][new_j].side == self.cells[i][j].side.opposite(),
                    ])
                    obstructed = False  # If path to cell is obstructed by other figures
                    if not (figure.type == FigureType.GOLD_KNIGHT and figure.state == 1):  # not knight
                        dy_sign = dy // abs(dy) if dy != 0 else 0
                        dx_sign = dx // abs(dx) if dx != 0 else 0
                        tmp_i, tmp_j = new_i - dy_sign, new_j - dx_sign
                        while (tmp_i, tmp_j) != (i, j):
                            if not self.is_empty_cell(tmp_i, tmp_j):
                                obstructed = True
                            tmp_i -= dy_sign
                            tmp_j -= dx_sign

                    if not_friendly_fire and not obstructed:
                        moves.append(Move(
                            array_destination=(new_i, new_j),
                            figure=figure,
                            array_origin=(i, j),
                        ))
        return moves

    def is_empty_cell(self, i: int, j: int) -> bool:
        return self.cells[i][j] is None

    def get_side_inventory(self, side: Side):
        if side == Side.BLACK:
            return self.inventory_black
        return self.inventory_white

    def copy(self):
        return Board(
            copy.deepcopy(self.cells),
            copy.deepcopy(self.inventory_black),
            copy.deepcopy(self.inventory_white),
            self.turn,
        )

    def print(self):
        for row in self.cells:
            for cell in row:
                if cell is None:
                    print(".   ", end="")
                else:
                    direction_chr = "^" if cell.side == Side.BLACK else "v"
                    figure_chr = cell.type.to_jp()[cell.state]
                    print(f"{figure_chr}{direction_chr} ", end="")
            print()

    def move_from_array_coords(self, origin: tuple[int, int], destination: tuple[int, int]):
        figure = self.cells[origin[0]][origin[1]]
        return Move(
            destination,
            figure,
            origin,
            is_drop=False,
        )