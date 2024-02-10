import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA device
    print("GPU is available")
else:
    device = torch.device("cpu")   # Use CPU
    print("GPU is not available, using CPU")


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to the range [-1, 1]
])

class DeepDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)     
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 10)        
        self.relu = nn.ReLU()

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)
        init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        # print("x size:", x.size())
        x = x.view(-1, 28*28) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

seed_number = 42
torch.manual_seed(seed_number)
np.random.seed(seed_number)

loss_func = nn.CrossEntropyLoss()
deepmodel_MINST = DeepDNN()
deepmodel_MINST.to(device)

learning_rate = 0.01
num_epochs = 100

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_dataset.targets = torch.tensor(np.random.randint(0, 10, (len(train_dataset)),))
# test_dataset.targets = torch.tensor(np.random.randint(0, 10, (len(test_dataset)),))
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

def training_process(model, trainloader, testloader, optimizer, num_epochs, train_loss_log, test_loss_log):
    for epoch in range(num_epochs):
        
        cur_loss = 0
        num_samples = 0
        for i, data in enumerate(trainloader, 0):
            X, labels = data
            X = X.to(device)
            labels = labels.to(device)

            outputs = model(X)

            loss = loss_func(outputs, labels)

        # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_loss += loss.item()
            num_samples += 1

            if (i) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        train_loss_log.append(cur_loss / num_samples)

        cur_loss = 0
        num_samples = 0

        for data in testloader:
            X, labels = data
            X = X.to(device)
            labels = labels.to(device)
            
            outputs = model(X)
            loss = loss_func(outputs, labels)
            cur_loss += loss.item()
            num_samples += 1

        test_loss_log.append(cur_loss / num_samples)

optimizer1 = torch.optim.SGD(deepmodel_MINST.parameters(), lr=learning_rate)
deep_MINST_loss_log = []
deep_MINST_test_loss_log = []
training_process(deepmodel_MINST, trainloader, testloader, optimizer1, num_epochs, deep_MINST_loss_log, deep_MINST_test_loss_log)

plt.plot(deep_MINST_loss_log, 'r', label='training loss')
plt.plot(deep_MINST_test_loss_log, 'g', label='test loss')
plt.title("training loss")
plt.xlabel("EPOCHS")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig('HW-3-1')
# plt.show()