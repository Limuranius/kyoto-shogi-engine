import engine
from Board import Board
from Figure import Figure, FigureType, Side
from Move import Move
import speed_analyzer
import bitboard

import pandas as pd
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)


def test1():
    board = Board.default_board()

    board = board.make_move(board.move_from_array_coords((4, 3), (3, 2)))
    board = board.make_move(board.move_from_array_coords((0, 0), (1, 0)))

    evaluator = engine.Evaluator()
    move_picker = engine.MovePicker(
        evaluator,
        max_depth=3,
    )

    best_move = move_picker.pick_best_move(board)
    print(best_move)

    print(board)


def tsume1():
    board_str = """
金金
 ． 金v 玉v 銀v とv 
飛v  ．  ．  ．  ． 
 ．  ． 金v  ．  ． 
 ．  ．  ．  ．  ． 
 ．  ． 玉^  ．  ． 

"""[1:]
    board = Board.from_str(board_str, Side.WHITE)
    print(board)

    evaluator = engine.Evaluator()
    move_picker = engine.MovePicker(
        evaluator,
        max_depth=3,
    )
    move = move_picker.pick_best_move(board)

    board = board.make_move(move)
    print(board)

    print(speed_analyzer.get_stats())


def interactive():
    board = Board.default_board()
    evaluator = engine.Evaluator()
    move_picker = engine.MovePicker(
        evaluator,
        max_depth=3,
    )
    print(board)

    while True:
        user_move = input("Enter move in format 'ij ij': ")
        i1 = int(user_move[0])
        j1 = int(user_move[1])
        i2 = int(user_move[3])
        j2 = int(user_move[4])
        user_move = board.move_from_array_coords((i1, j1), (i2, j2))
        board = board.make_move(user_move)
        print(board)

        bot_move = move_picker.pick_best_move(board)
        board = board.make_move(bot_move)
        print(board)


def bitboard_test():
    board_str = """
金金
 ． 金v 玉v 銀v とv 
飛v  ．  ．  ．  ． 
歩v  ． 金v  ．  ． 
 ．  ．  ．  ．  ． 
 ．  ． 玉^  ．  ． 

"""[1:]
    board = Board.from_str(board_str, Side.WHITE)
    bb = board.to_bitboard()
    bitboard.Bitarray5x5.print(bb[bitboard.ATTACKS_WHITE])

    print(bitboard.get_bitboard_moves(bb))


def bitboard_tsume1():
    board_str = """
金金
 ． 金v 玉v 銀v とv 
飛v  ．  ．  ．  ． 
 ．  ． 金v  ．  ． 
 ．  ．  ．  ．  ． 
 ．  ． 玉^  ．  ． 

"""[1:]
    board = Board.from_str(board_str, Side.WHITE)
    print(board)
    bb = board.to_bitboard()

    evaluator = bitboard.BitboardEvaluator()
    move_picker = bitboard.BitboardMovePicker(
        evaluator,
        max_depth=4,
    )
    move = move_picker.pick_best_move(bb)
    print(move)

    # board = board.make_move(move)
    # print(board)

    print(speed_analyzer.get_stats())



# interactive()
# test1()
# tsume1()
# bitboard_test()
bitboard_tsume1()