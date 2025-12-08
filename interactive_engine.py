import numpy as np

import bitboard
from Board import Board, Side
from Move import Move
from Figure import Figure, FigureType


board = Board.default_board()


def get_best_move(board: Board) -> tuple[Move, Board]:
    bb = board.to_bitboard()

    # bitboard.diagnostic(bb)

    evaluator = bitboard.BitboardEvaluator()
    move_picker = bitboard.BitboardMovePicker(evaluator, max_depth=5, max_time=30)
    best_move = move_picker.pick_batch_best_move(bb)
    is_drop = len(best_move) == 2
    if is_drop:
        new_bb = bitboard.make_drop_fast(bb, *best_move)
    else:
        new_bb = bitboard.make_move_fast(bb, *best_move)
    new_board = Board.from_bitboard(new_bb)

    bitboard_figure_to_figure = {
        bitboard.TOKIN_BLACK: Figure(type=FigureType.TOKIN_LANCE, side=Side.BLACK, state=0),
        bitboard.LANCE_BLACK: Figure(type=FigureType.TOKIN_LANCE, side=Side.BLACK, state=1),
        bitboard.SILVER_BLACK: Figure(type=FigureType.SILVER_BISHOP, side=Side.BLACK, state=0),
        bitboard.BISHOP_BLACK: Figure(type=FigureType.SILVER_BISHOP, side=Side.BLACK, state=1),
        bitboard.KING_BLACK: Figure(type=FigureType.KING, side=Side.BLACK, state=0),
        bitboard.GOLD_BLACK: Figure(type=FigureType.GOLD_KNIGHT, side=Side.BLACK, state=0),
        bitboard.KNIGHT_BLACK: Figure(type=FigureType.GOLD_KNIGHT, side=Side.BLACK, state=1),
        bitboard.PAWN_BLACK: Figure(type=FigureType.PAWN_ROOK, side=Side.BLACK, state=0),
        bitboard.ROOK_BLACK: Figure(type=FigureType.PAWN_ROOK, side=Side.BLACK, state=1),
        bitboard.TOKIN_WHITE: Figure(type=FigureType.TOKIN_LANCE, side=Side.WHITE, state=0),
        bitboard.LANCE_WHITE: Figure(type=FigureType.TOKIN_LANCE, side=Side.WHITE, state=1),
        bitboard.SILVER_WHITE: Figure(type=FigureType.SILVER_BISHOP, side=Side.WHITE, state=0),
        bitboard.BISHOP_WHITE: Figure(type=FigureType.SILVER_BISHOP, side=Side.WHITE, state=1),
        bitboard.KING_WHITE: Figure(type=FigureType.KING, side=Side.WHITE, state=0),
        bitboard.GOLD_WHITE: Figure(type=FigureType.GOLD_KNIGHT, side=Side.WHITE, state=0),
        bitboard.KNIGHT_WHITE: Figure(type=FigureType.GOLD_KNIGHT, side=Side.WHITE, state=1),
        bitboard.PAWN_WHITE: Figure(type=FigureType.PAWN_ROOK, side=Side.WHITE, state=0),
        bitboard.ROOK_WHITE: Figure(type=FigureType.PAWN_ROOK, side=Side.WHITE, state=1),
    }

    if is_drop:
        drop_pos_bit, figure_index = best_move
        drop_coord = bitboard.get_bit_coord(drop_pos_bit)
        figure = bitboard_figure_to_figure[figure_index]
        move = Move(drop_coord, figure, is_drop=True)
    else:
        start_pos_bit, end_pos_bit, figure_index = best_move
        start_coord = bitboard.get_bit_coord(start_pos_bit)
        end_coord = bitboard.get_bit_coord(end_pos_bit)
        figure = bitboard_figure_to_figure[figure_index]
        move = Move(end_coord, figure, start_coord, is_drop=False)
    return move, new_board

print("Kyoto shogi engine")
player_side = input("Choose side ('black' or 'white'): ")
player_side = {"black": Side.BLACK, "white": Side.WHITE}[player_side]

str_to_figure = {
    "tokin": Figure(type=FigureType.TOKIN_LANCE, side=player_side, state=0),
    "lance": Figure(type=FigureType.TOKIN_LANCE, side=player_side, state=1),
    "silver": Figure(type=FigureType.SILVER_BISHOP, side=player_side, state=0),
    "bishop": Figure(type=FigureType.SILVER_BISHOP, side=player_side, state=1),
    "king": Figure(type=FigureType.KING, side=player_side, state=0),
    "gold": Figure(type=FigureType.GOLD_KNIGHT, side=player_side, state=0),
    "knight": Figure(type=FigureType.GOLD_KNIGHT, side=player_side, state=1),
    "pawn": Figure(type=FigureType.PAWN_ROOK, side=player_side, state=0),
    "rook": Figure(type=FigureType.PAWN_ROOK, side=player_side, state=1),
}

board_history = [board]

while True:
    print(board)
    if board.turn == player_side:
        print("Enter your move. Two formats:")
        print("   1. For move type: i0 j0 i1 j1")
        print(f"   2. For drop type: i0 j0 figure - figure out of list {list(str_to_figure.keys())}")
        print("   3. 'cancel' - to cancel previous your and engine moves")
        inp = input("> ")
        inp = inp.strip().split(" ")
        if len(inp) == 4:
            i0, j0, i1, j1 = map(int, inp)
            move = board.move_from_array_coords(origin=(i0, j0), destination=(i1, j1))
            board = board.make_move(move)
            board_history.append(board)
        elif len(inp) == 3:
            i0, j0, figure = inp
            if figure not in str_to_figure:
                print("Wrong figure")
            i0 = int(i0)
            j0 = int(j0)
            move = Move(array_destination=(i0, j0), figure=str_to_figure[figure], is_drop=True)
            board = board.make_move(move)
            board_history.append(board)
        elif inp[0] == "cancel":
            board_history.pop()
            board_history.pop()
            board = board_history[-1]
        else:
            print("bruh")
    else:
        print("Engine turn")
        move, board = get_best_move(board)
        print(move)
        board_history.append(board)

