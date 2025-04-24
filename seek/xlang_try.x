import cantor through 'lrpc:1000'
from garnet import garnet
T = garnet.tensor()
garnet.cantor = cantor
modelPath = "C:/df/moe-16b/*.bin"
model = garnet.loadModel(modelPath)
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = model.tokenizer(text, return_tensors = "pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
print(inputs)
model_embed_tokens_weight = model["model.embed_tokens.weight"]

@T.fusion()
def ft(w):
    s= w* T.softmax ()
    return s

t1 = tensor([1,2,3]) 
st= ft(t1)
print('st:',st)

@T.fusion()
def fe(w,x):
    s_embed = w*T.gather()*x
    return s_embed
# s_graph = T.graph(s_embed)
# s_graph.run()
s_embed = fe(model_embed_tokens_weight, input_ids)
print("s_embed:",s_embed)
model_layers_0_input_layernorm_weight = model["model.layers.0.input_layernorm.weight"]
Y = model_embed_tokens_weight * model_layers_0_input_layernorm_weight
y_graph = T.graph(Y)
print("y_graph:",y_graph)
y_graph.run()
print("Y=",Y)
print("Done")

class MT:
    xx
    yy
    zz
    def MT():
        pass
    def add(x, y):
        this.xx = x
        this.yy = y

        this.zz = this.xx + this.yy
        return this.zz

x=tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
y=200
mt = MT()
z = mt.add(x,y)

t_g = T.graph(z)
t_g.run()
print(z)



inputs_filename='./data/inputs_data2.json'
fileObj = fs.File(inputs_filename,"r")
content = fileObj.read(fileObj.size)
fileObj.close()
print(content)
inputs = json.loads(content,normalize=True)
print(inputs)

# f = fs.File(filename,"r");
#   f_size = f.size;
#   if f_size >=0:
#     code = f.read(f_size)
#   f.close();

# s ="dasgfsgsdgsg"
# inputs = json.loads(s,normalize=True)
# inputs = json.loads(inputs_filename,normalize=True)
# t_g = T.graph(inputs)
# t_g.run()
# print(inputs)


print("*******************************************")
# add a number to a tensor
t1 = tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print("t1 =", t1)

t11 = 10 + t1
print(t11)
t_g = T.graph(t11)
t_g.run()
print(t_g)
print('t11:',t11)
print ("***********************************************************************")

#t1 = tensor([[1,2,5],[0,0,0], [1,2,1]])
t1 = tensor([[1,2,5],[7, 5, 9], [4,2,1]])
print("t1 =", t1)
#t2 = tensor([[-1,-2,-1],[0,0,0], [1,2,1]])
t2 = tensor([[1,2,3], [4,5,6], [7,8,9]])
print("t2 =", t2)

t11 = t1*T.conv2d()*t2
t_g = T.graph(t11)
t_g.run()
print("after graph run, t1*T.conv2d()*t2=", t11)
#expected sum is = [[-16, -24,-28,-23],[-24,-32,-32,-24],[-24,-32,-32,-24],[28,40,44,35]]  //same mode
#expected result t1*T.conv2d()*t2= Tensor(size=(6,6),[-1,-4,-8,-12,-11,-4,-5,-16,-24,-28,-23,-8,-8,-24,-32,-32,-24,-8,-8,-24,-32,-32,-24,-8,9,28,40,44,35,12,13,40,56,60,47,16])
print ("***********************************************************************")
#to minus a tensor to a tensor with different dimensions
t1 = tensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
print("t1 =", t1)
t2 = tensor([[1,2],[3,4]])
print("t2 =", t2)
import CpuTensor as T
t11 = t1-t2
print("before graph run, t1-t2=", t11)
t_g = T.graph(t11)
print(t_g)
t_g.run()
print("after graph run, t1-t2=", t11)

#expected t1-t2= Tensor(size=(3,2,2),[0,0,0,0,4,4,4,4,8,8,8,8])

print('Done.')