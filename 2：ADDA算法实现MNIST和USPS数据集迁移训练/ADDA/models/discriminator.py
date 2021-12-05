# ok
# ADDA 的鉴别器模型
from torch import nn


# 源域的鉴别器模型
class Discriminator(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        out = self.layer(input)
        return out
