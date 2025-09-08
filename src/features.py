import numpy as np
import chess

# ---- Board encoding (AlphaZero-lite): 12 piece planes + stm + castling(4) + parity = 18 channels ----

PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
]

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Return [C,8,8] float32 tensor."""
    planes = []

    # 12 planes for piece presence (white then black)
    for color in [chess.WHITE, chess.BLACK]:
        for p in PIECE_ORDER:
            bb = board.pieces(p, color)
            arr = np.zeros((8,8), dtype=np.float32)
            for sq in bb:
                r, c = divmod(sq, 8)
                arr[r, c] = 1.0
            planes.append(arr)

    # side-to-move
    stm = np.full((8,8), 1.0 if board.turn == chess.WHITE else 0.0, dtype=np.float32)
    planes.append(stm)

    # castling rights: wK, wQ, bK, bQ
    cast = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    for flag in cast:
        planes.append(np.full((8,8), 1.0 if flag else 0.0, dtype=np.float32))

    # move-count parity (or could scale fifty-move)
    parity = np.full((8,8), float(board.fullmove_number % 2), dtype=np.float32)
    planes.append(parity)

    x = np.stack(planes, axis=0)  # [C,8,8]
    return x

# ---- Action mapping (simple 64x64 = 4096). Promotions default to QUEEN. ----

ACTION_SIZE = 64 * 64

def move_to_index(move: chess.Move) -> int:
    """Map UCI move to 0..4095 via from*64 + to. Promotions collapse to same index."""
    return move.from_square * 64 + move.to_square

def index_to_move(board: chess.Board, idx: int) -> chess.Move | None:
    """Return a legal chess.Move for from/to; auto-queen if promotion is needed."""
    if idx < 0 or idx >= ACTION_SIZE:
        return None
    from_sq, to_sq = divmod(idx, 64)
    move = chess.Move(from_sq, to_sq)
    # if promotion required and not provided, use queen
    if move.promotion is None:
        # If moving a pawn to last rank, set promotion
        if board.piece_type_at(from_sq) == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if to_rank == 7 and board.turn == chess.WHITE:
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
            if to_rank == 0 and board.turn == chess.BLACK:
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    if move in board.legal_moves:
        return move
    return None

def legal_mask(board: chess.Board) -> np.ndarray:
    """Binary mask [4096] with ones for legal from/to pairs (promotion variants collapse)."""
    mask = np.zeros((ACTION_SIZE,), dtype=np.float32)
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        mask[idx] = 1.0
    return mask
