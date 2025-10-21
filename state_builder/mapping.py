from typing import Dict, Optional

def map_class_to_fen(cls_name: str, class_to_fen: Dict[str, str]) -> Optional[str]:
    """Returns FEN char for a detector class name with light normalization."""
    if cls_name is None:
        return None
    k = cls_name.strip()
    fen = class_to_fen.get(k)
    if fen is not None:
        return fen
    k_norm = k.replace("-", "_").lower()
    return class_to_fen.get(k_norm)