import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()#show the pictures by svg style, making them more clear

trans=transforms.ToTensor()#transform the pics into tensor in Pytorch,and scaling the pixel values from 0 to 1
mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
##download two dataset into assigned directory
##root:assign directory ; train=True:load the dataset ; transform=trans: apply Totensor() to dataset
## download=True if the dataset has not been downloaded,then download it.
def get_fashion_mnist_labels(labels):#fetch the  number of each class
    text_labels= ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):#a function for showing pictures
    #num_rows for the rownumber of pictures,and num_cols for the columns of pictures
    figsize=(num_cols*scale,num_rows*scale)# set the size of pics
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    #creat a big canvas
    axes=axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        #zip() is  to match the axes with imgs
        #enumerate() is to add index for each element
        if torch.is_tensor(img):
            ax.imshow(img.numpy())# tensor must transform into numpy and then can be displayed
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)#hide the x-axis
        ax.axes.get_yaxis().set_visible(False)#so does y=axis
        if titles:
            ax.set_title(titles[i])# show the title of pics
    return axes
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
#next(iter()) means get the first batch of data
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

batch_size=256

def get_dataloader_workers():
    return 4

train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())
# shuffle is to strengthen the generalization of models
timer=d2l.Timer()# create a counter
for X,y in train_iter:#do nothing but test the time of loading datas
    continue
print(f'{timer.stop():.2f} sec')# show it out

