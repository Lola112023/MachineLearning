import torch

x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[2.0],[4.0],[6.0]]) #self-defined some dataset

class LinearModel(torch.nn.Module):#it's a type of object-oriented programming style
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear= torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred= self.linear(x)
        return y_pred
    ##forward function is automatically called each time I use object LinearModel,which is model in this program

model= LinearModel()#create an object instance so that we can implement the object we define above

criterion = torch.nn.MSELoss(size_average=False)#choose a loss function ,mean square error.

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)# choose an optimizer so in which we choose SGD here
# the learning rate should not be too large,should be small.

for epoch in range(100):
    y_pred =model(x_data)# plug the data we assign into our model,having an output y
    loss=criterion(y_pred,y_data)# calculate the loss by loss function we define
    print(epoch,loss.item())
## the reason why we should use item() to take the value of what we need is that the tensor of pytorch
    ##have many values in there,including gradiant and so on.
    loss.backward()#calculate the back propagation
    optimizer.step()# update the parameters
    optimizer.zero_grad() #clean up the parameters so that we can have our next step.

print('w=',model.linear.weight.item())# take out the weight in our linear model by member function item
print('b=',model.linear.bias.item())# take out the bias in our linear model by member function item

x_test=torch.Tensor([[4.0]])#we define a new input to test the accuracy of our model
y_test=model(x_test)#predict a y by our model
print('y_pred',y_test.data)#print out the prediction which should be 8 actually
