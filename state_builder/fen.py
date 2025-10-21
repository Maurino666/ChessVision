from typing import List, Optional

def build_fen_from_grid(grid: List[List[Optional[str]]]) -> str:
    """
    Builds a full FEN string from an oriented 8x8 grid.
    Assumes grid[0] is rank 8, grid[7] is rank 1, and entries are FEN chars or None.
    """
    rows = []
    for r in range(8):
        empty = 0
        out = ""
        for c in range(8):
            ch = grid[r][c]
            if ch is None:
                empty += 1
            else:
                if empty > 0:
                    out += str(empty)
                    empty = 0
                out += ch
        if empty > 0:
            out += str(empty)
        rows.append(out)
    board = "/".join(rows)
    return f"{board} w - - 0 1"

def algebraic_square_from_row_col(row: int, col: int) -> str:
    """Converts oriented grid indices to algebraic notation. row 0 = rank 8, col 0 = file 'a'."""
    files = "abcdefgh"
    file_ch = files[col]
    rank_ch = str(8 - row)
    return f"{file_ch}{rank_ch}"

