"""
A classification head that predicts each level of S2 cells separately, and uses the
predictions/logits as inputs to the next layer.

>>> from . import s2cell_mapping
>>> from . import label_mapping
>>> mapping = s2cell_mapping.S2CellMapping(label_mapping.LabelMapping([
...     "8085", "808584", "80857c" # San Francisco
... ]))
>>> h2c = HierS2Classifier(512, mapping)
>>> x = torch.zeros(7, 512)
>>> h2c(x).shape
torch.Size([7, 3])
"""

import torch
from torch import nn
import torch.nn.functional as F

import timm

class HierS2Classifier(nn.Module):
    def __init__(self, in_features, s2cell_mapping):
        super().__init__()

        self.s2cell_mapping = s2cell_mapping

        # Create hierarchical layers
        self.hier = []
        for level in range(0, s2cell_mapping.max_cell_level + 1):
            layer_in = in_features
            layer_in += sum(l.out_features for (_, l) in self.hier)

            layer_outs = len(s2cell_mapping.tokens_by_level[level])
            if layer_outs > 0:
                layer = nn.Linear(layer_in, layer_outs)
                self.hier.append((level, layer))

    def forward(self, x):
        logits = torch.zeros(x.shape[0], len(self.s2cell_mapping.label_mapping))
        for level, layer in self.hier:
            y_partial = layer(x)
            for ix, label in enumerate(self.s2cell_mapping.labels_by_level(level)):
                logits[:, label] = y_partial[:, ix]
            x = torch.cat([x, y_partial], dim=1)
            x = F.relu(x)

        return logits


class HierS2ClassifierHead(nn.Module):
    def __init__(self, in_features, s2cell_mapping):
        super().__init__()

        self.global_pool = timm.layers.SelectAdaptivePool2d(pool_type='avg')
        self.norm = timm.layers.Norm2d(num_channels=in_features, eps=1e-5)
        self.flatten = nn.Flatten(1)

        self.hier = HierS2Classifier(in_features, s2cell_mapping)


    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.hier(x)
        return x


if __name__ == "__main__":
    import doctest
    doctest.testmod()