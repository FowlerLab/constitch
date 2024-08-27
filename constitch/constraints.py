from typing import Optional
import dataclasses


@dataclasses.dataclass
class Constraint:
    dx: int
    dy: int
    score: Optional[float] = None
    overlap: Optional[float] = None
    modeled: bool = False
    error: int = 0

