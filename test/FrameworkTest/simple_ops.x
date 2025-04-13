import cantor through 'lrpc:1000'
from garnet import garnet

# Set cantor in garnet
garnet.cantor = cantor

# Create a tensor handler
T = garnet.tensor()

@T.fusion()
def calc_test_simple():
    t1 = tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])  # 4×3 matrix
    t2 = tensor([[10,20,30,40],[50,60,70,80],[90,100,110,120]])  # 3×4 matrix

    # Perform matrix multiplication
    # t1 is 4×3 and t2 is 3×4, so multiplication is valid
    Z = t1 * t2
    Z = Z+10
    Z = Z*100
    return Z

result = calc_test_simple()
print("Done")