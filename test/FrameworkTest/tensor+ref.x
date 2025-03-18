import cantor through 'lrpc:1000'
from garnet import garnet

def test(x):
    return x

k = test(100)
garnet.cantor = cantor

# Create a tensor handler
T = garnet.tensor()

@T.fusion()
def ref_param_test(x):
    print("in ref_param_test:",x)
    return x

t2 = ref_param_test(100)
print("Done")