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

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

class DeepDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)        
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, 32)
        self.fc8 = nn.Linear(32, 10)        
        self.relu = nn.ReLU()

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)
        init.xavier_uniform_(self.fc5.weight)
        init.xavier_uniform_(self.fc6.weight)
        init.xavier_uniform_(self.fc7.weight)
        init.xavier_uniform_(self.fc8.weight)

    def forward(self, x):
        # print("x size:", x.size())
        x = x.view(-1, 28*28) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        return x

class ShallowDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 88)
        self.fc2 = nn.Linear(88, 10)
        self.relu = nn.ReLU()
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 28*28) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

seed_number = 42
torch.manual_seed(seed_number)
np.random.seed(seed_number)

loss_func = nn.CrossEntropyLoss()
deepmodel_MINST = DeepDNN()
shallowmodel_MINST = ShallowDNN()
deepmodel_MINST.to(device)
shallowmodel_MINST.to(device)

print("deep model parameters: ", sum(p.numel() for p in deepmodel_MINST.parameters() if p.requires_grad))
print("shallow model parameters: ", sum(p.numel() for p in shallowmodel_MINST.parameters() if p.requires_grad))


learning_rate = 0.02
num_epochs = 40

def training_process(model, trainloader, testloader, optimizer, num_epochs, loss_log, acc_log, func_mode=1):
    for epoch in range(num_epochs):

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

            loss_log.append(loss.item())
            if (i) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        
        correct_count = 0
        total_count = 0
        for data in testloader:
            X, labels = data
            X = X.to(device)
            labels = labels.to(device)
            outputs = model(X)
            _, predictions = torch.max(outputs.data, 1)
            total_count += labels.size(0)
            correct_count += (predictions == labels).sum().item()
            acc_log.append(100 * correct_count / total_count)

optimizer1 = torch.optim.SGD(deepmodel_MINST.parameters(), lr=learning_rate)
deep_MINST_loss_log = []
deep_MINST_acc_log = []
training_process(deepmodel_MINST, trainloader, testloader, optimizer1, num_epochs, deep_MINST_loss_log, deep_MINST_acc_log)


optimizer2 = torch.optim.SGD(shallowmodel_MINST.parameters(), lr=learning_rate)
shallow_MINST_loss_log = []
shallow_MINST_acc_log = []
training_process(shallowmodel_MINST, trainloader, testloader, optimizer2, num_epochs, shallow_MINST_loss_log, shallow_MINST_acc_log)

plt.plot(deep_MINST_loss_log, 'r', label='deep model')
plt.plot(shallow_MINST_loss_log, 'g', label='shallow model')
plt.title("training loss")
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig('HW-1-5')
plt.close()

plt.plot(deep_MINST_acc_log, 'r', label='deep model')
plt.plot(shallow_MINST_acc_log, 'g', label='shallow model')
plt.title("training accurate")
plt.xlabel("steps")
plt.ylabel("accurate")
plt.legend(loc="lower right")
plt.savefig('HW-1-6')
plt.close()
