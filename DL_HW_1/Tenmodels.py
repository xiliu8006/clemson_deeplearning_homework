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
    def __init__(self, layers_params):
        super().__init__()
        self.fc1 = nn.Linear(784, layers_params[0])
        self.fc2 = nn.Linear(layers_params[0], layers_params[1])     
        self.fc3 = nn.Linear(layers_params[1], layers_params[2])
        self.fc4 = nn.Linear(layers_params[2], layers_params[3])
        self.fc5 = nn.Linear(layers_params[3], 10)        
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

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

seed_number = 42
torch.manual_seed(seed_number)
np.random.seed(seed_number)

loss_func = nn.CrossEntropyLoss()

def training_process(model, trainloader, testloader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        
        cur_loss = 0
        num_samples = 0
        correct_count = 0
        total_count = 0
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

            _, predictions = torch.max(outputs.data, 1)
            total_count += labels.size(0)
            correct_count += (predictions == labels).sum().item()

            if (i) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        if epoch == (num_epochs - 1):
            train_loss = cur_loss / num_samples
            train_acc = 100 * correct_count / total_count

        cur_loss = 0
        num_samples = 0
        correct_count = 0
        total_count = 0
        for data in testloader:
            X, labels = data
            X = X.to(device)
            labels = labels.to(device)
            
            outputs = model(X)
            loss = loss_func(outputs, labels)
            cur_loss += loss.item()
            num_samples += 1
            _, predictions = torch.max(outputs.data, 1)
            total_count += labels.size(0)
            correct_count += (predictions == labels).sum().item()
        if epoch == (num_epochs - 1):
            test_loss = cur_loss / num_samples
            test_acc = 100 * correct_count / total_count

    return train_loss, train_acc, test_loss, test_acc
        


Ten_layer_params = [
    [64, 64, 32, 32],
    [64, 32, 32, 32],
    [64, 24, 24, 24],
    [32, 32, 32, 32],
    [32, 32, 32, 16],
    [32, 24, 24, 16],
    [24, 24, 24, 24],
    [24, 24, 16, 16],
    [24, 16, 16, 16],
    [16, 16, 16, 8],
    [16, 8, 8, 8],
]

all_train_loss = []
all_train_acc = []
all_test_loss = []
all_test_acc = []
all_total_params = []

for layer_params in Ten_layer_params:

    deepmodel_MINST = DeepDNN(layer_params)
    deepmodel_MINST.to(device)

    learning_rate = 0.01
    num_epochs = 200


    optimizer1 = torch.optim.SGD(deepmodel_MINST.parameters(), lr=learning_rate)
    train_loss, train_acc, test_loss, test_acc = training_process(deepmodel_MINST, trainloader, testloader, optimizer1, num_epochs)
    total_params = sum(p.numel() for p in deepmodel_MINST.parameters() if p.requires_grad)
    all_train_loss.append(train_loss)
    all_train_acc.append(train_acc)
    all_test_loss.append(test_loss)
    all_test_acc.append(test_acc)
    all_total_params.append(total_params)

plt.plot(all_total_params, all_train_loss, 'r*', label='train')
plt.plot(all_total_params, all_test_loss, 'g^', label='test')
plt.title("loss with different parameter number")
plt.xlabel("model's parameter number")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig('HW-3-2')
plt.close()

plt.plot(all_total_params, all_train_acc, 'r*',  label='train')
plt.plot(all_total_params, all_test_acc, 'g^', label='test')
plt.title("loss with different parameter number")
plt.xlabel("model's parameter number")
plt.ylabel("accuracy")
plt.legend(loc="lower right")
plt.savefig('HW-3-3')
# plt.show()