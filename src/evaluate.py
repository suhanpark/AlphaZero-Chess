import chess
import numpy as np
import torch
from .network import PolicyValueNet
from .mcts import MCTS
from .features import board_to_tensor, legal_mask

@torch.no_grad()
def eval_fn_from_weights(ckpt_path, C=18, device="cuda"):
    model = PolicyValueNet(C).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    def _fn(batch_np):
        x = torch.from_numpy(batch_np).to(device)
        p, v = model(x)
        return p.cpu().numpy(), v.cpu().numpy()
    return _fn

def play_match(evalA, evalB, sims=200, games=20):
    """A plays White on even games, swap colors each game."""
    results = []
    for g in range(games):
        board = chess.Board()
        eval_w = evalA if g % 2 == 0 else evalB
        eval_b = evalB if g % 2 == 0 else evalA
        mcts_w = MCTS(eval_w, sims=sims)
        mcts_b = MCTS(eval_b, sims=sims)
        while not board.is_game_over():
            if board.turn:
                _, visits = mcts_w.search(board, add_dirichlet=False)
            else:
                _, visits = mcts_b.search(board, add_dirichlet=False)
            a = int(np.argmax(visits))
            from_sq, to_sq = divmod(a, 64)
            mv = None
            for leg in board.legal_moves:
                if leg.from_square == from_sq and leg.to_square == to_sq:
                    mv = leg; break
            if mv is None:
                mv = list(board.legal_moves)[0]
            board.push(mv)
        results.append(board.result())
    return results
