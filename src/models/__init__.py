from src.encoders import GINEncoder    # side-effect: registers "gin" in ENCODERS
from src.nn import DINOHead            # side-effect: registers "dino" in HEADS
from src.core.model import BaseModel
from .graphdino import GraphDINO
