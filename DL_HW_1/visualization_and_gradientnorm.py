import torch
import numpy as np
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from sklearn.decomposition import PCA

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
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=8, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

class ShallowDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 28*28) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ShallowDNN_func(nn.Module):
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

loss_func = nn.CrossEntropyLoss()
shallowmodel_func = ShallowDNN_func()

learning_rate = 0.02
num_epochs = 30

def training_process(model, trainloader, testloader, optimizer, num_epochs, loss_log, grad_log, weights_log, func_mode=1):
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
                grad_sum = 0.0
                for p in model.parameters():
                    grad = 0.0
                    if p.grad is not None:
                        grad = (p.grad.cpu().data.numpy() ** 2).sum()
                    grad_sum += grad
                grad_norm = grad_sum ** 0.5
                grad_log.append(grad_norm)
        
        
        if (epoch % 3) == 0:
            weights = copy.deepcopy(model.state_dict())
            weights_log.append(weights)

total_weights = {}
N_trains = 8
for i in range(N_trains):
    shallowmodel_MINST = ShallowDNN()
    shallowmodel_MINST.to(device)
    optimizer = torch.optim.SGD(shallowmodel_MINST.parameters(), lr=learning_rate)
    shallow_MINST_loss_log = []
    shallow_MINST_grad_log = []
    weights_log = []
    training_process(shallowmodel_MINST, trainloader, testloader, optimizer, num_epochs, shallow_MINST_loss_log, shallow_MINST_grad_log, weights_log)
    total_weights[i] = weights_log

# for the first layers visualizaitons
N_samples = len(total_weights[0])
origin_data = []
for i in range(N_trains):
    for j in range(N_samples):
        origin_data.append(total_weights[i][j]['fc1.weight'].cpu().numpy().flatten())
origin_data = np.array(origin_data)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(origin_data)
# pca_data = pca(origin_data, 2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
for i in range(8):
    plt.scatter(pca_data[i * N_samples: (i+1) * N_samples, 0], pca_data[i * N_samples: (i+1) * N_samples, 1], color=colors[i], label=f'The {i+1}th of train')
plt.title(" Only layer 1")
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.legend(loc="upper right")
plt.savefig('HW-2-1')
plt.close()

N_samples = len(total_weights[0])
origin_data = []
for i in range(N_trains):
    for j in range(N_samples):
        origin_data.append(np.concatenate((total_weights[i][j]['fc1.weight'].cpu().numpy().flatten(),\
                                           total_weights[i][j]['fc2.weight'].cpu().numpy().flatten())))
origin_data = np.array(origin_data)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(origin_data)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
for i in range(8):
    plt.scatter(pca_data[i * N_samples: (i+1) * N_samples, 0], pca_data[i * N_samples: (i+1) * N_samples, 1], color=colors[i], label=f'The {i+1}th of train')
plt.title(" Whole model")
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.legend(loc="upper right")
plt.savefig('HW-2-2')
plt.close()


def training_process1(model, optimizer, num_epochs, loss_log, grad_log, pca_log, func_mode=1):
    loss_func = nn.MSELoss()
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
        
        weights = []
        if (epoch % 3) == 0:
            for param in model.parameters():
                weights.append(param.data.view(-1).numpy())
            weights = np.concatenate(weights)
            pca_log.append(weights)
        
        grad_sum = 0
        for p in model.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy() ** 2).sum()
            grad_sum += grad
        gradNorm = grad_sum ** 0.5
        grad_log.append(gradNorm)


optimizer1 = torch.optim.SGD(shallowmodel_func.parameters(), lr=0.03)
shallow_func_loss_log = []
shallow_func_grad_log = []
pca_log = []
num_epochs = 20000
training_process1(shallowmodel_func, optimizer1, num_epochs, shallow_func_loss_log, shallow_func_grad_log, pca_log)
data_numpy = np.array(pca_log)

shallowmodel_MINST = ShallowDNN()
shallowmodel_MINST.to(device)
optimizer2 = torch.optim.SGD(shallowmodel_MINST.parameters(), lr=learning_rate)
shallow_MINST_loss_log = []
shallow_MINST_grad_log = []
weights_log = []
num_epochs = 40
training_process(shallowmodel_MINST, trainloader, testloader, optimizer2, num_epochs, shallow_MINST_loss_log, shallow_MINST_grad_log, weights_log)

plt.plot(shallow_MINST_grad_log, 'g', label='shallow-MNIST')
plt.title("grad minst")
plt.xlabel("steps")
plt.ylabel("grad")
plt.legend(loc="upper right")
plt.savefig('HW-2-3')
plt.close()

plt.plot(shallow_MINST_loss_log, 'g', label='shallow-MNIST')
plt.title("loss minst")
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig('HW-2-4')
plt.close()

plt.plot(shallow_func_grad_log, 'g', label='shallow-func')
plt.title("grad func1")
plt.xlabel("steps")
plt.ylabel("grad")
plt.legend(loc="upper right")
plt.savefig('HW-2-5')
plt.close()

plt.plot(shallow_func_loss_log, 'g', label='shallow-func')
plt.title("loss func1")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig('HW-2-6')
plt.close()


# class DeepDNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(1, 5)
#         self.fc2 = nn.Linear(5, 10)
#         self.fc3 = nn.Linear(10, 10)
#         self.fc4 = nn.Linear(10, 10)        
#         self.fc5 = nn.Linear(10, 5)
#         self.fc6 = nn.Linear(5, 1)        
#         self.relu = nn.ReLU()

#         init.xavier_uniform_(self.fc1.weight)
#         init.xavier_uniform_(self.fc2.weight)
#         init.xavier_uniform_(self.fc3.weight)
#         init.xavier_uniform_(self.fc4.weight)
#         init.xavier_uniform_(self.fc5.weight)
#         init.xavier_uniform_(self.fc6.weight)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.relu(self.fc4(x))
#         x = self.relu(self.fc5(x))
#         x = self.fc6(x)
#         return x

# from torch.autograd.functional import hessian
# def compute_hessian(model, criterion, X, labels):
#     outputs = model(X.view(X.shape[0], -1))
#     loss = criterion(outputs, labels)
#     print(model.parameters)
#     hessian_matrix = hessian(loss, model.parameters())
#     return hessian_matrix

# for i in range(10):
#     model = DeepDNN()
#     optimizer1 = torch.optim.SGD(model.parameters(), lr=0.005)
#     shallow_func_loss_log = []
#     shallow_func_grad_log = []
#     pca_log = []
#     num_epochs = 200
#     training_process1(model, optimizer1, num_epochs, shallow_func_loss_log, shallow_func_grad_log, pca_log)
#     X = torch.rand(100, 1)
#     hessian_matrix = hessian(model, X)
#     eigenvalues = torch.eig(hessian_matrix).eigenvalues
#     print("Eigenvalues:", eigenvalues)
#     data_numpy = np.array(pca_log)