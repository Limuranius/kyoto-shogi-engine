from Board import Board
from Figure import Side
from Move import Move
from engine.Evaluator import Evaluator
import tqdm


class MovePicker:
    max_depth: int
    evaluator: Evaluator

    def __init__(
            self,
            evaluator: Evaluator,
            max_depth: int,
    ):
        self.evaluator = evaluator
        self.max_depth = max_depth

    def pick_best_move(self, board: Board) -> Move:
        move_evals = []
        for move in tqdm.tqdm(list(board.yield_all_moves())):
            move_evals.append((move, self.recursive_evaluate(board.make_move(move), depth=1)))
        func = self.direction_function(board.turn)
        return func(move_evals, key=lambda x: x[1])[0]

    def recursive_evaluate(self, board: Board, depth: int) -> float:
        if depth == self.max_depth:
            return self.evaluator.evaluate_board(board)
        else:
            evals = []
            for move in list(board.yield_all_moves()):
                evals.append(self.recursive_evaluate(board.make_move(move), depth=depth + 1))
            func = self.direction_function(board.turn)
            return func(evals)

    def direction_function(self, side: Side):
        func = {Side.BLACK: max, Side.WHITE: min}
        return func[side]

