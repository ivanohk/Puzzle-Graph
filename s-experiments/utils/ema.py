import numpy as np

# Implement Exponential Moving Average 
class EMA:
    def __init__(self, alpha, epochs) -> None:
        super().__init__()
        self.alpha = alpha
        self.steps = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        
        alpha = 1 - (1 - self.alpha) * (np.cos(np.pi * self.steps / self.total_steps) + 1) / 2.0

        self.steps += 1

        ema = alpha * old + (1 - alpha) * new

        return ema