import numpy as np
import onnxruntime as ort
import chess
from .mcts import MCTS
from .features import board_to_tensor, legal_mask, index_to_move, ACTION_SIZE

class OnnxEvaluator:
    def __init__(self, onnx_path="az_chess.onnx"):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]  # ["policy_logits","value"]

    def __call__(self, batch_np):
        if batch_np.dtype != np.float32:
            batch_np = batch_np.astype(np.float32)
        outs = self.sess.run(self.out_names, {self.in_name: batch_np})
        policy_logits, value = outs
        return policy_logits, value.squeeze(-1)

def play_cli(onnx_path="az_chess.onnx", sims=200):
    eval_fn = OnnxEvaluator(onnx_path)
    mcts = MCTS(eval_fn, sims=sims, dirichlet_eps=0.0)  # no noise for play
    board = chess.Board()
    print(board)
    while not board.is_game_over():
        # Human move
        uci = input("Your move (e.g., e2e4): ").strip()
        try:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                raise ValueError()
            board.push(mv)
        except Exception:
            print("Illegal or bad format. Try again.")
            continue
        print(board)
        if board.is_game_over():
            break

        # AI move via MCTS+ONNX
        _, visits = mcts.search(board, add_dirichlet=False)
        a = int(np.argmax(visits))
        mv = index_to_move(board, a)
        if mv is None:
            # fallback: pick best legal by policy directly
            x = board_to_tensor(board)[None,...]
            logits, _ = eval_fn(x)
            mask = legal_mask(board)
            logits[0][mask == 0] = -1e9
            a = int(np.argmax(logits[0]))
            mv = index_to_move(board, a)
        print("AI:", mv.uci())
        board.push(mv)
        print(board)

    print("Game over:", board.result())

if __name__ == "__main__":
    play_cli()
