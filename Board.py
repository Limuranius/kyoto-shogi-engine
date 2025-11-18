from __future__ import annotations

import collections
import copy
from collections import defaultdict
from typing import Generator

import speed_analyzer
from Figure import Figure, Side, FigureType
from Move import Move
import bitboard

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
            turn: Side = Side.BLACK,
    ):
        self.cells = cells
        self.inventory_black = inventory_black
        self.inventory_white = inventory_white
        self.turn = turn

    @classmethod
    def empty_board(cls):
        cells = [
            [None] * 5,
            [None] * 5,
            [None] * 5,
            [None] * 5,
            [None] * 5,
        ]

        return Board(
            cells=cells,
            inventory_black=collections.defaultdict(int),
            inventory_white=collections.defaultdict(int),
            turn=Side.BLACK
        )

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
            inventory_white=collections.defaultdict(int),
            turn=Side.BLACK
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

    @speed_analyzer.add_to_watchlist
    def make_move(self, move: Move) -> Board:
        """Makes move and returns new version of board"""
        new_board = self.copy()
        i, j = move.array_destination
        inv = new_board.get_side_inventory(self.turn)
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

    @speed_analyzer.add_to_watchlist
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
            for destination in self.get_cell_destinations(i, j, include_friendly_fire=False):
                moves.append(Move(
                    array_destination=destination,
                    figure=figure,
                    array_origin=(i, j),
                ))

        return moves

    def get_cell_destinations(
            self,
            i: int, j: int,
            include_friendly_fire: bool = False
    ) -> Generator[tuple[int, int], None, None]:
        """
        Yields all possible destinations that a piece at cell (i, j) can go to
        if :include_friendly_fire is True then also returns destinations with friendly figures
        """
        if self.is_empty_cell(i, j):
            return

        figure = self.cells[i][j]
        for dy, dx in figure.get_moves():
            new_i, new_j = i + dy, j + dx
            if 0 <= new_i < 5 and 0 <= new_j < 5:  # in bounds
                not_friendly_fire = (
                        include_friendly_fire
                        or self.is_empty_cell(new_i, new_j)
                        or self.cells[new_i][new_j].side == self.cells[i][j].side.opposite()
                )
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
                    yield new_i, new_j

    def is_empty_cell(self, i: int, j: int) -> bool:
        return self.cells[i][j] is None

    def get_side_inventory(self, side: Side):
        if side == Side.BLACK:
            return self.inventory_black
        return self.inventory_white

    @speed_analyzer.add_to_watchlist
    def copy(self):
        return Board(
            copy.deepcopy(self.cells),
            copy.deepcopy(self.inventory_black),
            copy.deepcopy(self.inventory_white),
            self.turn,
        )

    def to_str(self) -> str:
        result = ""

        # print white inventory
        for fig_type in self.inventory_white:
            count = self.inventory_white[fig_type]
            result += fig_type.to_jp()[0] * count
        result += "\n"

        for row in self.cells:
            for cell in row:
                if cell is None:
                    result += " ． "
                else:
                    direction_chr = "^" if cell.side == Side.BLACK else "v"
                    figure_chr = cell.type.to_jp()[cell.state]
                    result += f"{figure_chr}{direction_chr} "
            result += "\n"

        # print black inventory
        for fig_type in self.inventory_black:
            count = self.inventory_black[fig_type]
            result += fig_type.to_jp()[0] * count

        return result

    def __str__(self):
        return self.to_str()

    def move_from_array_coords(self, origin: tuple[int, int], destination: tuple[int, int]):
        figure = self.cells[origin[0]][origin[1]]
        return Move(
            destination,
            figure,
            origin,
            is_drop=False,
        )

    @speed_analyzer.add_to_watchlist
    def get_king_position(self, side: Side) -> tuple[int, int]:
        for i in range(5):
            for j in range(5):
                cell = self.cells[i][j]
                if (
                        cell is not None
                        and cell.type == FigureType.KING
                        and cell.side == side
                ):
                    return i, j

    def yield_all_moves(self):
        for i in range(5):
            for j in range(5):
                for move in self.get_cell_moves(i, j):
                    yield move

    @classmethod
    def from_str(cls, str_board: str, side_to_move: Side) -> Board:
        """
        Parses board from string. Example of input:
            金金銀飛
             ． 金v 玉v 銀v とv
            飛v  ．  ．  ．  ．
             ．  ．  ．  ．  ．
             ．  ． 桂^  ．  ．
            と^ 銀^ 玉^  ． 歩^
            桂金歩

        First and last lines show inventories. Any character of figures can be used.
        For example 桂 and 金 will be treated same
        """
        rows = str_board.split("\n")

        # parsing board
        board_rows = rows[1: -1]
        cells = []
        chr_to_side = {"^": Side.BLACK, "v": Side.WHITE}
        for row in board_rows:
            row_cells = []
            for i in range(0, len(row), 3):
                cell_str = row[i: i + 3]
                if cell_str == " ． ":
                    figure = None
                else:
                    figure = Figure.from_jp(cell_str[0], chr_to_side[cell_str[1]])
                row_cells.append(figure)
            cells.append(row_cells)

        # parsing_inventories
        inv_white_str = rows[0]
        inv_black_str = rows[-1]
        inv_white = defaultdict(int)
        inv_black = defaultdict(int)
        for ch in inv_white_str:
            figure = Figure.from_jp(ch)
            inv_white[figure.type] += 1
        for ch in inv_black_str:
            figure = Figure.from_jp(ch)
            inv_black[figure.type] += 1

        return Board(
            cells=cells,
            inventory_black=inv_black,
            inventory_white=inv_white,
            turn=side_to_move,
        )

    def __hash__(self):
        return hash(self.to_str())

    def get_figure_positions(self, figure: Figure) -> list[tuple[int, int]]:
        coords = []
        for i in range(5):
            for j in range(5):
                if self.cells[i][j] == figure:
                    coords.append((i, j))
        return coords

    def to_bitboard(self) -> bitboard.Bitboard:
        bb = bitboard.get_empty_bitboard()

        board_figures = [
            (bitboard.TOKIN_BLACK, Figure(type=FigureType.TOKIN_LANCE, side=Side.BLACK, state=0)),
            (bitboard.LANCE_BLACK, Figure(type=FigureType.TOKIN_LANCE, side=Side.BLACK, state=1)),
            (bitboard.SILVER_BLACK, Figure(type=FigureType.SILVER_BISHOP, side=Side.BLACK, state=0)),
            (bitboard.BISHOP_BLACK, Figure(type=FigureType.SILVER_BISHOP, side=Side.BLACK, state=1)),
            (bitboard.KING_BLACK, Figure(type=FigureType.KING, side=Side.BLACK, state=0)),
            (bitboard.GOLD_BLACK, Figure(type=FigureType.GOLD_KNIGHT, side=Side.BLACK, state=0)),
            (bitboard.KNIGHT_BLACK, Figure(type=FigureType.GOLD_KNIGHT, side=Side.BLACK, state=1)),
            (bitboard.PAWN_BLACK, Figure(type=FigureType.PAWN_ROOK, side=Side.BLACK, state=0)),
            (bitboard.ROOK_BLACK, Figure(type=FigureType.PAWN_ROOK, side=Side.BLACK, state=1)),
            (bitboard.TOKIN_WHITE, Figure(type=FigureType.TOKIN_LANCE, side=Side.WHITE, state=0)),
            (bitboard.LANCE_WHITE, Figure(type=FigureType.TOKIN_LANCE, side=Side.WHITE, state=1)),
            (bitboard.SILVER_WHITE, Figure(type=FigureType.SILVER_BISHOP, side=Side.WHITE, state=0)),
            (bitboard.BISHOP_WHITE, Figure(type=FigureType.SILVER_BISHOP, side=Side.WHITE, state=1)),
            (bitboard.KING_WHITE, Figure(type=FigureType.KING, side=Side.WHITE, state=0)),
            (bitboard.GOLD_WHITE, Figure(type=FigureType.GOLD_KNIGHT, side=Side.WHITE, state=0)),
            (bitboard.KNIGHT_WHITE, Figure(type=FigureType.GOLD_KNIGHT, side=Side.WHITE, state=1)),
            (bitboard.PAWN_WHITE, Figure(type=FigureType.PAWN_ROOK, side=Side.WHITE, state=0)),
            (bitboard.ROOK_WHITE, Figure(type=FigureType.PAWN_ROOK, side=Side.WHITE, state=1)),
        ]
        for bitboard_figure_index, figure in board_figures:
            coords = self.get_figure_positions(figure)
            bb[bitboard_figure_index] = bitboard.position_mask_from_coordinates(coords)

        inv_figures_black = [
            (bitboard.TOKIN_BLACK, FigureType.TOKIN_LANCE),
            (bitboard.SILVER_BLACK, FigureType.SILVER_BISHOP),
            (bitboard.KING_BLACK, FigureType.KING),
            (bitboard.GOLD_BLACK, FigureType.GOLD_KNIGHT),
            (bitboard.PAWN_BLACK, FigureType.PAWN_ROOK),
        ]
        inv_figures_white = [
            (bitboard.TOKIN_WHITE, FigureType.TOKIN_LANCE),
            (bitboard.SILVER_WHITE, FigureType.SILVER_BISHOP),
            (bitboard.KING_WHITE, FigureType.KING),
            (bitboard.GOLD_WHITE, FigureType.GOLD_KNIGHT),
            (bitboard.PAWN_WHITE, FigureType.PAWN_ROOK),
        ]
        for bitboard_figure_index, figure_type in inv_figures_black:
            count = self.inventory_black.get(figure_type, 0)
            for _ in range(count):
                bitboard.increase_inventory_count(bb, bitboard_figure_index)
        for bitboard_figure_index, figure_type in inv_figures_white:
            count = self.inventory_white.get(figure_type, 0)
            for _ in range(count):
                bitboard.increase_inventory_count(bb, bitboard_figure_index)

        bb[bitboard.IS_BLACK_TURN] = self.turn == Side.BLACK
        bitboard.update_masks(bb)
        return bb

    @classmethod
    def from_bitboard(cls, bb: bitboard.Bitboard) -> Board:
        board = Board.empty_board()

        board_figures = [
            (bitboard.TOKIN_BLACK, Figure(type=FigureType.TOKIN_LANCE, side=Side.BLACK, state=0)),
            (bitboard.LANCE_BLACK, Figure(type=FigureType.TOKIN_LANCE, side=Side.BLACK, state=1)),
            (bitboard.SILVER_BLACK, Figure(type=FigureType.SILVER_BISHOP, side=Side.BLACK, state=0)),
            (bitboard.BISHOP_BLACK, Figure(type=FigureType.SILVER_BISHOP, side=Side.BLACK, state=1)),
            (bitboard.KING_BLACK, Figure(type=FigureType.KING, side=Side.BLACK, state=0)),
            (bitboard.GOLD_BLACK, Figure(type=FigureType.GOLD_KNIGHT, side=Side.BLACK, state=0)),
            (bitboard.KNIGHT_BLACK, Figure(type=FigureType.GOLD_KNIGHT, side=Side.BLACK, state=1)),
            (bitboard.PAWN_BLACK, Figure(type=FigureType.PAWN_ROOK, side=Side.BLACK, state=0)),
            (bitboard.ROOK_BLACK, Figure(type=FigureType.PAWN_ROOK, side=Side.BLACK, state=1)),
            (bitboard.TOKIN_WHITE, Figure(type=FigureType.TOKIN_LANCE, side=Side.WHITE, state=0)),
            (bitboard.LANCE_WHITE, Figure(type=FigureType.TOKIN_LANCE, side=Side.WHITE, state=1)),
            (bitboard.SILVER_WHITE, Figure(type=FigureType.SILVER_BISHOP, side=Side.WHITE, state=0)),
            (bitboard.BISHOP_WHITE, Figure(type=FigureType.SILVER_BISHOP, side=Side.WHITE, state=1)),
            (bitboard.KING_WHITE, Figure(type=FigureType.KING, side=Side.WHITE, state=0)),
            (bitboard.GOLD_WHITE, Figure(type=FigureType.GOLD_KNIGHT, side=Side.WHITE, state=0)),
            (bitboard.KNIGHT_WHITE, Figure(type=FigureType.GOLD_KNIGHT, side=Side.WHITE, state=1)),
            (bitboard.PAWN_WHITE, Figure(type=FigureType.PAWN_ROOK, side=Side.WHITE, state=0)),
            (bitboard.ROOK_WHITE, Figure(type=FigureType.PAWN_ROOK, side=Side.WHITE, state=1)),
        ]
        for bitboard_figure_index, figure in board_figures:
            coords = bitboard.get_bits_coords(bb[bitboard_figure_index])
            for i, j in coords:
                board.cells[i][j] = figure

        inv_figures_black = [
            (bitboard.TOKIN_BLACK, FigureType.TOKIN_LANCE),
            (bitboard.SILVER_BLACK, FigureType.SILVER_BISHOP),
            (bitboard.KING_BLACK, FigureType.KING),
            (bitboard.GOLD_BLACK, FigureType.GOLD_KNIGHT),
            (bitboard.PAWN_BLACK, FigureType.PAWN_ROOK),
        ]
        inv_figures_white = [
            (bitboard.TOKIN_WHITE, FigureType.TOKIN_LANCE),
            (bitboard.SILVER_WHITE, FigureType.SILVER_BISHOP),
            (bitboard.KING_WHITE, FigureType.KING),
            (bitboard.GOLD_WHITE, FigureType.GOLD_KNIGHT),
            (bitboard.PAWN_WHITE, FigureType.PAWN_ROOK),
        ]
        for bitboard_figure_index, figure_type in inv_figures_black:
            board.inventory_black[figure_type] = bitboard.get_inventory_count(bb, bitboard_figure_index)
        for bitboard_figure_index, figure_type in inv_figures_white:
            board.inventory_white[figure_type] = bitboard.get_inventory_count(bb, bitboard_figure_index)

        board.turn = Side.BLACK if bb[bitboard.IS_BLACK_TURN] else Side.WHITE
        return board