import math
import numpy as np
import chess
from dataclasses import dataclass, field
from .features import board_to_tensor, legal_mask, index_to_move, move_to_index, ACTION_SIZE

@dataclass
class EdgeStats:
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    P: float = 0.0

@dataclass
class Node:
    board: chess.Board
    to_play: bool  # True=white, False=black
    prior: float = 0.0
    edges: dict = field(default_factory=dict)  # action idx -> EdgeStats
    expanded: bool = False
    value: float | None = None

class MCTS:
    def __init__(self, model_eval, c_puct=1.2, dirichlet_eps=0.25, dirichlet_alpha=0.3, sims=400):
        """
        model_eval: callable([N,C,8,8]) -> (policy_logits [N,4096], values [N])
        """
        self.eval = model_eval
        self.c_puct = c_puct
        self.dir_eps = dirichlet_eps
        self.dir_alpha = dirichlet_alpha
        self.sims = sims

    def search(self, board: chess.Board, add_dirichlet=True):
        root = Node(board.copy(stack=False), board.turn)
        self._expand(root)

        # root Dirichlet noise
        if add_dirichlet:
            legal_idxs = list(root.edges.keys())
            noise = np.random.dirichlet([self.dir_alpha] * len(legal_idxs))
            for a, n in zip(legal_idxs, noise):
                root.edges[a].P = (1 - self.dir_eps) * root.edges[a].P + self.dir_eps * n

        for _ in range(self.sims):
            self._simulate(root)

        # Extract visit counts
        visits = np.zeros((ACTION_SIZE,), dtype=np.float32)
        for a, e in root.edges.items():
            visits[a] = e.N
        return root, visits

    def _simulate(self, node: Node):
        path = [node]
        # SELECTION
        while node.expanded:
            a = self._select_action(node)
            mv = index_to_move(node.board, a)
            node = node.edges[a].child
            if node is None:
                # create child on demand
                next_board = path[-1].board.copy(stack=True)
                next_board.push(mv)
                child = Node(next_board, next_board.turn, prior=0.0)
                node = child
                path[-1].edges[a].child = child
            path.append(node)

        # EVALUATE & EXPAND (unless terminal)
        value = self._evaluate_and_expand(node)

        # BACKUP (flip sign per ply)
        for n in reversed(path):
            if n is path[-1]:
                pass
            parent = path[path.index(n)-1] if n is not path[0] else None
        v = value
        # backup from leaf to root
        for n in reversed(path[:-1]):
            parent = n
            child = path[path.index(n)+1]
            # find action from parent->child
            action = None
            for a, e in parent.edges.items():
                if e.child is child:
                    action = a
                    break
            e = parent.edges[action]
            e.N += 1
            e.W += v
            e.Q = e.W / e.N
            v = -v  # flip perspective

    def _select_action(self, node: Node) -> int:
        sqrt_N = math.sqrt(sum(e.N for e in node.edges.values()) + 1e-8)
        best_a, best_score = None, -1e9
        for a, e in node.edges.items():
            u = self.c_puct * e.P * sqrt_N / (1 + e.N)
            score = e.Q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _expand(self, node: Node):
        if node.board.is_game_over():
            node.expanded = False
            node.value = self._terminal_value(node.board)
            return

        # NN eval
        x = board_to_tensor(node.board)[None, ...]
        p_logits, v = self.eval(x)  # numpy outputs
        p = self._policy_from_logits(p_logits[0], node.board)
        node.value = float(v[0])

        # init edges
        node.edges = {}
        for a, prior in p.items():
            node.edges[a] = EdgeStats(N=0, W=0.0, Q=0.0, P=prior)
            node.edges[a].child = None
        node.expanded = True

    def _evaluate_and_expand(self, node: Node) -> float:
        if node.board.is_game_over():
            return self._terminal_value(node.board)
        self._expand(node)
        return node.value

    def _policy_from_logits(self, logits: np.ndarray, board: chess.Board) -> dict[int, float]:
        mask = legal_mask(board)  # [4096]
        logits = logits.copy()
        logits[mask == 0] = -1e9
        # softmax
        x = logits - logits.max()
        probs = np.exp(x)
        Z = probs.sum()
        if Z <= 0:
            # fallback uniform over legal
            idxs = np.flatnonzero(mask)
            return {int(i): 1.0/len(idxs) for i in idxs}
        probs /= Z
        # return sparse dict (only legal)
        return {int(i): float(probs[i]) for i in np.flatnonzero(mask)}

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        res = board.result()
        if res == "1-0": return 1.0
        if res == "0-1": return -1.0
        return 0.0
