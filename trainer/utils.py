from typing import Dict

import torch.nn as nn
from ignite.engine import Engine


def print_model_summary(engine: Engine, model: nn.Module) -> None:
    try:
        import torchsummary
        device = engine.state.device
        s, _ = torchsummary.summary_string(model, (3, 96, 96), device=device)
        print(s)
    except:
        print(model)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of parameters: {}'.format(n))
