from .encoder import GINEncoder      # side-effect: registers "gin" in ENCODERS
from .head import DINOHead            # side-effect: registers "dino" in HEADS
from .model_types import BaseModel, GraphDINO
