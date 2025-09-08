import numpy as np
import random
import chess
from tqdm import trange
from .mcts import MCTS
from .features import board_to_tensor, move_to_index, legal_mask

def play_selfplay_games(model_eval, n_games=50, sims=400, temp_moves=30,
                        dirichlet=(0.25, 0.3), seed=42):
    """
    model_eval: callable([N,C,8,8]) -> (policy_logits [N,4096], values [N]) returning numpy
    Returns list of (state_tensor, pi (4096,), z) tuples.
    """
    random.seed(seed); np.random.seed(seed)
    games_data = []
    for _ in trange(n_games, desc="Self-play"):
        board = chess.Board()
        trajectory = []  # list of (state, pi, player)
        mcts = MCTS(model_eval, dirichlet_eps=dirichlet[0], dirichlet_alpha=dirichlet[1], sims=sims)
        ply = 0

        while not board.is_game_over():
            root, visits = mcts.search(board, add_dirichlet=(ply < temp_moves))
            pi = visits / (visits.sum() + 1e-8)

            # temperature: sample for first temp_moves plies, then argmax
            if ply < temp_moves:
                legal_idxs = np.flatnonzero(visits > 0)
                probs = pi[legal_idxs]
                probs /= probs.sum()
                a = int(np.random.choice(legal_idxs, p=probs))
            else:
                a = int(np.argmax(visits))

            x = board_to_tensor(board)
            trajectory.append((x, pi, board.turn))
            mv = None
            # Convert to a legal move (already ensured by MCTS)
            from_sq, to_sq = divmod(a, 64)
            for leg in board.legal_moves:
                if leg.from_square == from_sq and leg.to_square == to_sq:
                    mv = leg; break
            if mv is None:  # fallback, shouldn't happen
                mv = list(board.legal_moves)[0]
            board.push(mv)
            ply += 1

        # game result
        res = board.result()
        z_white = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0.0)

        # assign outcomes per player-to-move at each state
        for x, pi, player_white in trajectory:
            z = z_white if player_white else -z_white
            games_data.append((x.astype(np.float32), pi.astype(np.float32), np.float32(z)))
    return games_data

class ReplayBuffer:
    def __init__(self, capacity=300_000):
        self.capacity = capacity
        self.states, self.pis, self.zs = [], [], []

    def add_many(self, samples):
        for s, p, z in samples:
            self.states.append(s); self.pis.append(p); self.zs.append(z)
        # trim
        if len(self.states) > self.capacity:
            cut = len(self.states) - self.capacity
            self.states = self.states[cut:]
            self.pis    = self.pis[cut:]
            self.zs     = self.zs[cut:]

    def sample(self, batch_size):
        idx = np.random.choice(len(self.states), size=batch_size, replace=False)
        S = np.stack([self.states[i] for i in idx]).astype(np.float32)
        P = np.stack([self.pis[i]    for i in idx]).astype(np.float32)
        Z = np.stack([self.zs[i]     for i in idx]).astype(np.float32)
        return S, P, Z

    def __len__(self):
        return len(self.states)
