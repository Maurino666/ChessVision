from typing import List, Optional
import numpy as np

def rotate_grid_180(grid: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """Returns a new grid rotated by 180 degrees."""
    return [list(reversed(row)) for row in reversed(grid)]


def rotate_grid_90_cw(grid: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """Returns a new grid rotated by 90 degrees clockwise."""
    size = 8
    return [[grid[size - 1 - r2][c2] for r2 in range(size)] for c2 in range(size)]


def rotate_grid_90_ccw(grid: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """Returns a new grid rotated by 90 degrees counter-clockwise."""
    size = 8
    return [[grid[r2][size - 1 - c2] for r2 in range(size)] for c2 in range(size)]


def rotate_conf(conf: np.ndarray, white_side: str) -> np.ndarray:
    """Rotates the 8x8 confidence array to match the oriented grid."""
    if white_side == "south":
        return conf
    if white_side == "north":
        return np.rot90(conf, 2)
    if white_side == "west":
        return np.rot90(conf, 1)   # CCW
    if white_side == "east":
        return np.rot90(conf, -1)  # CW
    raise ValueError("Invalid white_side")
