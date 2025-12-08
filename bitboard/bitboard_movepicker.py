import numpy as np
import tqdm
import time

from . import BitboardBatch
from .Bitboard import *
from .bitboard_evaluator import BitboardEvaluator, BLACK, WHITE


class BitboardMovePicker:
    max_depth: int
    evaluator: BitboardEvaluator
    max_time: float  # max execution time in seconds

    def __init__(
            self,
            evaluator: BitboardEvaluator,
            max_depth: int,
            max_time: float,
    ):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.max_time = max_time

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
            if len(evals) == 0:
                return 0.0
            else:
                return func(evals)

    def direction_function(self, side: int):
        func = {BLACK: max, WHITE: min}
        return func[side]

    """Batch functionality"""

    def direction_function_batch(self, side: int):
        func = {BLACK: np.max, WHITE: np.min}
        return func[side]

    def pick_batch_best_move(self, bitboard: Bitboard):
        self.start_time = time.perf_counter()

        batch = BitboardBatch.bitboard_to_batch(bitboard)  # batch of one board

        moves_and_drops = BitboardBatch.get_bitboards_moves(batch)

        # approximating move count
        b = batch.copy()
        b[IS_BLACK_TURN] = ~b[IS_BLACK_TURN].astype(bool)
        other_moves_and_drops = BitboardBatch.get_bitboards_moves(b)
        n_moves_and_drops = (len(moves_and_drops[0][0]) + len(moves_and_drops[0][1])) * (len(other_moves_and_drops[0][0]) + len(other_moves_and_drops[0][1]))
        # print(n_moves_and_drops)
        if n_moves_and_drops <= 350:
            print("Switching to depth 5")
            self.max_depth = 5
        else:
            print("Switching to depth 4")
            self.max_depth = 4

        new_batch = BitboardBatch.make_batch_moves_and_drops(  # batch of all moves from one board
            batch,
            moves_and_drops,
            concat=True,
        )

        self.pbar = tqdm.tqdm(desc=f"Evaluating boards (depth {self.max_depth})")
        evals = self.recursive_batch_evaluate(
            batch=new_batch,
            depth=1
        )
        self.pbar.close()

        if bitboard[IS_BLACK_TURN]:
            best_i = evals.argmax()
        else:
            best_i = evals.argmin()

        if best_i < len(moves_and_drops[0][0]):
            return moves_and_drops[0][0][best_i]
        else:
            return moves_and_drops[0][1][best_i - len(moves_and_drops[0][0])]

    def recursive_batch_evaluate(self, batch: BitboardBatch, depth: int) -> np.ndarray:
        n_boards = batch.shape[1]
        self.pbar.update(n_boards)
        if time.perf_counter() - self.start_time > self.max_time:  # out of time
            return np.zeros(n_boards)
        if n_boards == 0:
            return np.zeros(1)
        if depth == self.max_depth:
            return self.evaluator.evaluate_board(batch)
        else:
            evals = np.zeros(n_boards, dtype=float)
            batch_moves_and_drops = BitboardBatch.get_bitboards_moves(batch)
            n_moves = [len(moves) for moves, drops in batch_moves_and_drops]
            n_drops = [len(drops) for moves, drops in batch_moves_and_drops]
            func = self.direction_function_batch(batch[IS_BLACK_TURN, 0])

            new_batch = BitboardBatch.make_batch_moves_and_drops(
                    batch,
                    batch_moves_and_drops,
                    concat=True,
            )
            new_batch_evals = self.recursive_batch_evaluate(
                batch=new_batch,
                depth=depth + 1
            )

            i_move = 0
            i_drop = sum(n_moves)
            for i, (n_mv, n_dp) in enumerate(zip(n_moves, n_drops)):
                if n_mv + n_dp != 0:
                    evals[i] = func(np.concat([
                        new_batch_evals[i_move: i_move + n_mv],
                        new_batch_evals[i_drop: i_drop + n_dp],
                    ]))
                i_move += n_mv
                i_drop += n_dp

            # for i, new_batch in enumerate(BitboardBatch.make_batch_moves_and_drops(
            #         batch,
            #         batch_moves_and_drops,
            #         concat=False,
            # )):
            #     evals[i] = func(self.recursive_batch_evaluate(
            #         batch=new_batch,
            #         depth=depth + 1
            #     ))
            return evals
