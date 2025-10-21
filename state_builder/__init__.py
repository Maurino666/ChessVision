# board_state_builder/__init__.py
# Package initialization for the Board State Builder.
# Exposes only the public build_board_state() façade.
# All internal modules (geometry, assignment, orientation, fen, etc.)
# remain internal and should not be imported directly by the application layer.

from .builder import build_board_state

__all__ = ["build_board_state"]