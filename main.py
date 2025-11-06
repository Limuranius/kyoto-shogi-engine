from Board import Board
from Figure import Figure, FigureType
from Move import Move


board = Board.default_board()

move = board.move_from_array_coords((0, 0), (1, 0))
print(move)
board = board.make_move(move)
board = board.make_move(board.move_from_array_coords((4, 3), (3, 2)))

board.print()