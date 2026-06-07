from garnet import garnet
T = garnet.tensor()

def forward(a, b):
    return a * T.matmul() * b
