import gymnasium as gym
import numpy as np
import chess
from .features import board_to_tensor, legal_mask, index_to_move, ACTION_SIZE

class ChessAZEnv(gym.Env):
    """Minimal Gymnasium-like single-agent wrapper around python-chess for AlphaZero self-play."""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self):
        super().__init__()
        self.board = chess.Board()
        C = board_to_tensor(self.board).shape[0]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(C,8,8), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(ACTION_SIZE)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        obs = board_to_tensor(self.board)
        info = {"legal_mask": legal_mask(self.board)}
        return obs, info

    def step(self, action: int):
        """Action is 0..4095 (from/to); returns (obs, reward, terminated, truncated, info)"""
        mv = index_to_move(self.board, action)
        if mv is None:
            # illegal -> large negative reward & terminate
            return board_to_tensor(self.board), -1.0, True, False, {"illegal": True}
        self.board.push(mv)
        terminated = self.board.is_game_over()
        reward = 0.0
        if terminated:
            res = self.board.result()  # "1-0", "0-1", "1/2-1/2"
            reward = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0.0)
        obs = board_to_tensor(self.board)
        info = {"legal_mask": legal_mask(self.board)}
        return obs, reward, terminated, False, info

    def render(self):
        return str(self.board)

    @property
    def legal_actions(self):
        mask = legal_mask(self.board)
        return np.flatnonzero(mask).tolist()

    def clone(self):
        e = ChessAZEnv()
        e.board = self.board.copy(stack=True)
        return e
