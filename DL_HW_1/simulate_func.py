import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

class DeepDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)        
        self.fc5 = nn.Linear(10, 5)
        self.fc6 = nn.Linear(5, 1)        
        self.relu = nn.ReLU()

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)
        init.xavier_uniform_(self.fc5.weight)
        init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class ShallowDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 120)
        self.fc2 = nn.Linear(120, 1)
        self.relu = nn.ReLU()
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

seed_number = 42
torch.manual_seed(seed_number)
np.random.seed(seed_number)

deepmodel_func1 = DeepDNN()
shallowmodel_func1 = ShallowDNN()
print("deep model parameters: ", sum(p.numel() for p in deepmodel_func1.parameters() if p.requires_grad))
print("shallow model parameters: ", sum(p.numel() for p in shallowmodel_func1.parameters() if p.requires_grad))

deepmodel_func2 = DeepDNN()
shallowmodel_func2 = ShallowDNN()
learning_rate = 0.02
num_epochs = 20000
loss_func = nn.MSELoss()


def training_process(model, optimizer, num_epochs, loss_log, func_mode=1):
    X = torch.rand(100, 1)
    if func_mode == 1:
        Y_func = np.sin(5*np.pi * X) / (5*np.pi * X + 1e-5)
    else:
        Y_func = np.sign(np.sin(5*np.pi * X))

    for epoch in range(num_epochs):
        outputs = model(X)
        loss = loss_func(outputs, Y_func)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


optimizer1 = torch.optim.SGD(deepmodel_func1.parameters(), lr=learning_rate)
deep_func1_loss_log = []
training_process(deepmodel_func1, optimizer1, num_epochs, deep_func1_loss_log)


optimizer2 = torch.optim.SGD(shallowmodel_func1.parameters(), lr=learning_rate)
shallow_func1_loss_log = []
training_process(shallowmodel_func1, optimizer2, num_epochs, shallow_func1_loss_log)

#-------------------------------------------function 2 ---------------------------------------

optimizer3 = torch.optim.SGD(deepmodel_func2.parameters(), lr=learning_rate)
deep_func2_loss_log = []
training_process(deepmodel_func2, optimizer3, num_epochs, deep_func2_loss_log, 2)

optimizer4 = torch.optim.SGD(shallowmodel_func2.parameters(), lr=learning_rate)
shallow_func2_loss_log = []
training_process(shallowmodel_func2, optimizer4, num_epochs, shallow_func2_loss_log, 2)

# visulazation
plt.plot(deep_func1_loss_log[30:], 'r', label='deep-1')
plt.plot(shallow_func1_loss_log[30:], 'g', label='shallow-1')
plt.title("training loss")
plt.xlabel("EPOCHS")
plt.ylabel("MSE loss")
plt.legend(loc="upper right")
plt.savefig('HW-1-1')
plt.close()


plt.plot(deep_func2_loss_log[30:], 'r', label='deep-1')
plt.plot(shallow_func2_loss_log[30:], 'g', label='shallow-1')
plt.title("training loss")
plt.xlabel("EPOCHS")
plt.ylabel("MSE loss")
plt.legend(loc="upper right")
plt.savefig('HW-1-2')
plt.close()


X = torch.linspace(0, 1, 100).unsqueeze(1)
Y_func1 = np.sin(5*np.pi * X) / (5*np.pi * X + 1e-5)
Y_func2 = np.sign(np.sin(5*np.pi * X))

deepmodel_output1 = deepmodel_func1(X)
deepmodel_output2 = deepmodel_func2(X)

shallowmodel_output1 = shallowmodel_func1(X)
shallowmodel_output2 = shallowmodel_func2(X)

plt.plot(deepmodel_output1.tolist(), 'r.', linewidth=1, linestyle='-', label='deep-1')
plt.plot(shallowmodel_output1.tolist(), 'g.', linewidth=1, linestyle='-', label='shallow-1')
plt.plot(Y_func1.tolist(), 'b.', linewidth=1, linestyle='-', label='gt')
plt.title("predictions and ground truth")
plt.xlabel("input")
plt.ylabel("output")
plt.legend(loc="lower right")
plt.savefig('HW-1-3')
plt.close()

plt.plot(deepmodel_output2.tolist(), 'r.', linewidth=1, linestyle='-', label='deep-1')
plt.plot(shallowmodel_output2.tolist(), 'g.', linewidth=1, linestyle='-', label='shallow-1')
plt.plot(Y_func2.tolist(), 'b.', linewidth=1, linestyle='-', label='gt')
plt.title("predictions and ground truth")
plt.xlabel("input")
plt.ylabel("output")
plt.legend(loc="lower right")
plt.savefig('HW-1-4')
plt.close()