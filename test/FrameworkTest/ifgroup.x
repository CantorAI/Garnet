import cantor through 'lrpc:1000'
from garnet import garnet


xyz =100
t = tensor(xyz)
# Set cantor in garnet
garnet.cantor = cantor

# Create a tensor handler
T = garnet.tensor()

@T.fusion()
def calc_test(x,y,z):
    # Initialize tensor data properly
    t1 = tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])  # 4×3 matrix
    t2 = tensor([[10,20,30,40],[50,60,70,80],[90,100,110,120]])  # 3×4 matrix

    # Perform matrix multiplication
    # t1 is 4×3 and t2 is 3×4, so multiplication is valid
    Z = t1 * t2
    if x > 9 and x <100:
        if x >10:
            x =x+1
            Z = Z-1
        if z >100:
            Z = Z+x
        elif z >90:
            Z = Z+100
        else:
            x = x+100
            Z = Z-x
    elif x >20:
        Z = Z+y
    else:
        Z = Z-z
    Z = Z+1
    # Create and run the computation graph
    # y_graph = T.graph(Z)
    # print("Graph:",y_graph)
    # y_graph.run()
    return Z

result = calc_test(10.0,20.0,30.0)
result2 = calc_test(1110.0,120.0,130.0)
print("Done")