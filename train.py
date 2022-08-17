import torch
import torch.nn as nn
import torch.utils.data as Data
from net import MyLeNet_5
import torch.optim as optim
from torchvision import transforms, datasets
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#Data to Tensor
data_transform = transforms.Compose(
    [transforms.ToTensor()]
)

#MNIST import
train_dataset=datasets.MNIST(root='./data', train=True ,transform=data_transform,download=True)
train_dataloader=Data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)

#load test
test_dataset=datasets.MNIST(root='./data', train=False ,transform=data_transform,download=True)
test_dataloader=Data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=False)

#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#import GPU
model=MyLeNet_5().to(device)

#Loss Function
loss_fn=nn.CrossEntropyLoss()

#optimizer
optimizer=optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)

# Learning Rate,防止抖动太大,每十轮变成原来的0.1
Lr_scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

#train
def train(dataloader,model,loss_fn,optimizer):
    loss,current,n = 0.0, 0.0 ,0
    for batch,(X,y) in enumerate(dataloader):
        #forward
        X,y=X.to(device),y.to(device)
        output = model(X)
        cur_loss = loss_fn(output,y)
        _, pred = torch.max(output,axis=1)

        cur_acc=torch.sum(y==pred)/output.shape[0]

        #backward
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item() #累加本批次损失
        current += cur_acc.item()
        n+=1
    
    print("train_loss:"+str(loss/n))
    print("train_acc:"+str(current/n))

#验证
def val(dataloader,model,loss_fn,i):
    model.eval()
    loss,current,n = 0.0, 0.0 ,0
    with torch.no_grad():
        for batch,(X,y) in enumerate(dataloader):
            X,y=X.to(device),y.to(device)
            output = model(X)
            cur_loss = loss_fn(output,y)
            _, pred = torch.max(output,axis=1)
            cur_acc=torch.sum(y==pred)/output.shape[0]
            loss += cur_loss.item() #累加本批次损失
            current += cur_acc.item()
            n+=1
    writer.add_scalar("Loss/train",cur_loss , i)
    writer.add_scalar("Acc/train", cur_acc, i)
    print("val_loss:"+str(loss/n))
    print("val_acc:"+str(current/n))

    return current/n
    
#start train
epoch =50
min_acc= 0
for i in range(epoch):
    print(f'round{i+1}\n-------------')
    train(train_dataloader,model,loss_fn,optimizer)
    a = val(test_dataloader,model,loss_fn,i)
    #save best model
    if a >min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc=a
        print('save best model')
        torch.save(model.state_dict(),'save_model/best_model.pth')
print('Finished Training')
writer.flush()
writer.close()
