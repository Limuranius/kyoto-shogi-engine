from .bitboard_evaluator import BitboardEvaluator, BLACK, WHITE
from .Bitboard import *
import tqdm


class BitboardMovePicker:
    max_depth: int
    evaluator: BitboardEvaluator

    def __init__(
            self,
            evaluator: BitboardEvaluator,
            max_depth: int,
    ):
        self.evaluator = evaluator
        self.max_depth = max_depth

    def pick_best_move(self, bitboard: Bitboard):
        move_evals = []

        moves, drops = get_bitboard_moves(bitboard)

        pbar = tqdm.tqdm(total=len(moves) + len(drops))
        for i in range(len(moves)):
            move_evals.append((
                moves[i],
                self.recursive_evaluate(
                    make_move_fast(
                        bitboard,
                        prev_pos_bit=moves[i, 0],
                        new_pos_bit=moves[i, 1],
                        figure_index=moves[i, 2],
                    ),
                    depth=1,
                )
            ))
            pbar.update(1)
        for i in range(len(drops)):
            move_evals.append((
                drops[i],
                self.recursive_evaluate(
                    make_drop_fast(
                        bitboard,
                        drop_pos_bit=drops[i, 0],
                        figure_index=drops[i, 1],
                    ),
                    depth=1,
                )
            ))
            pbar.update(1)

        func = self.direction_function(bitboard[IS_BLACK_TURN])
        return func(move_evals, key=lambda x: x[1])[0]

    def recursive_evaluate(self, bitboard: Bitboard, depth: int) -> float:
        if depth == self.max_depth:
            return self.evaluator.evaluate_board(bitboard)
        else:
            evals = []
            moves, drops = get_bitboard_moves(bitboard)
            for i in range(len(moves)):
                evals.append(self.recursive_evaluate(
                    make_move_fast(
                        bitboard,
                        prev_pos_bit=moves[i, 0],
                        new_pos_bit=moves[i, 1],
                        figure_index=moves[i, 2],
                    ),
                    depth=depth + 1,
                ))
            for i in range(len(drops)):
                evals.append(self.recursive_evaluate(
                    make_drop_fast(
                        bitboard,
                        drop_pos_bit=drops[i, 0],
                        figure_index=drops[i, 1],
                    ),
                    depth=depth + 1,
                ))

            func = self.direction_function(bitboard[IS_BLACK_TURN])
            return func(evals)

    def direction_function(self, side: int):
        func = {BLACK: max, WHITE: min}
        return func[side]
