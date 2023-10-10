from dataclasses import dataclass

@dataclass(frozen=True)
class _Config:
    MAX_TIMESTEP = 300
    BATCH_SIZE = 128
    IMG_SIZE = 64
    NUM_EPOCHS = 100

diffusion_config = _Config()