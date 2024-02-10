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
seed_number = 42
torch.manual_seed(seed_number)
np.random.seed(seed_number)

loss_func = nn.CrossEntropyLoss()

def training_process(model, optimizer, num_epochs, batch_size):
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(num_epochs):
        
        cur_loss = 0
        num_samples = 0
        correct_count = 0
        total_count = 0
        frobenius_norm = 0
        for i, data in enumerate(trainloader, 0):
            X, labels = data
            X = X.to(device)
            labels = labels.to(device)
            if epoch == (num_epochs - 1):
                X.requires_grad_(True)
            outputs = model(X)
            loss = loss_func(outputs, labels)

        # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch == (num_epochs - 1):
                cur_loss += loss.item()
                num_samples += 1

                _, predictions = torch.max(outputs.data, 1)
                total_count += labels.size(0)
                correct_count += (predictions == labels).sum().item()
                # print("input grad: ", X.grad)
                frobenius_norm += torch.norm(X.grad, p='fro').sum() / X.shape[0]
                
    
            if (i) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        if epoch == (num_epochs - 1):
            train_loss = cur_loss / num_samples
            train_acc = 100 * correct_count / total_count
            train_sensitivity = frobenius_norm.detach().cpu().numpy() / total_count

        cur_loss = 0
        num_samples = 0
        correct_count = 0
        total_count = 0
        frobenius_norm = 0
        if epoch == (num_epochs - 1):
            for data in testloader:
                X, labels = data
                X = X.to(device)
                X.requires_grad_(True)
                labels = labels.to(device)
                
                outputs = model(X)
                loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                cur_loss += loss.item()
                num_samples += 1
                _, predictions = torch.max(outputs.data, 1)
                total_count += labels.size(0)
                correct_count += (predictions == labels).sum().item()
                frobenius_norm += torch.norm(X.grad, p='fro').sum() / X.shape[0]


        # if epoch == (num_epochs - 1):
            test_loss = cur_loss / num_samples
            test_acc = 100 * correct_count / total_count
            test_sensitivity = frobenius_norm.detach().cpu().numpy() / total_count


    return train_loss, train_acc, test_loss, test_acc, train_sensitivity, test_sensitivity
        
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


deepmodel_MINST_1 = DeepDNN(Ten_layer_params[1])
deepmodel_MINST_1.to(device)

deepmodel_MINST_2 = DeepDNN(Ten_layer_params[1])
deepmodel_MINST_2.to(device)

def merge_models(model1, model2, ratio):
    merged_model = DeepDNN(Ten_layer_params[1])  # Create an instance of the same type as model1
    for param1, param2, new_param in zip(model1.parameters(), model2.parameters(), merged_model.parameters()):
        new_param.data = ratio * param1.data + (1 - ratio) * param2.data
    return merged_model

deepmodel_MINST_3 = merge_models(deepmodel_MINST_1, deepmodel_MINST_2, 0.5)

learning_rate = 0.01
num_epochs = 30

# optimizer1 = torch.optim.SGD(deepmodel_MINST_1.parameters(), lr=learning_rate)
# train_loss, train_acc, test_loss, test_acc, _, _ = training_process(deepmodel_MINST_1, optimizer1, num_epochs, 64)
# optimizer2 = torch.optim.SGD(deepmodel_MINST_2.parameters(), lr=learning_rate)
# train_loss, train_acc, test_loss, test_acc, _, _ = training_process(deepmodel_MINST_2, optimizer2, num_epochs, 1024)

# alphas = torch.linspace(0.05, 0.95, 10)

# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# all_train_loss = []
# all_train_acc = []
# all_test_loss = []
# all_test_acc = []
# all_alpha = []

# for alpha in alphas:
#     deepmodel_MINST_3 = merge_models(deepmodel_MINST_1, deepmodel_MINST_2, alpha)
#     all_alpha.append(alpha)
#     cur_loss = 0
#     num_samples = 0
#     correct_count = 0
#     total_count = 0
#     for i, data in enumerate(trainloader, 0):
#         X, labels = data
#         X = X.to(device)
#         labels = labels.to(device)
#         outputs = deepmodel_MINST_3(X)
#         loss = loss_func(outputs, labels)
#         cur_loss += loss.item()
#         num_samples += 1

#         _, predictions = torch.max(outputs.data, 1)
#         total_count += labels.size(0)
#         correct_count += (predictions == labels).sum().item()
#     train_loss = cur_loss / num_samples
#     train_acc = 100 * correct_count / total_count
#     all_train_loss.append(train_loss)
#     all_train_acc.append(train_acc)
#     print("train loss and acc: ", train_loss, train_acc)


#     cur_loss = 0
#     num_samples = 0
#     correct_count = 0
#     total_count = 0
#     for i, data in enumerate(testloader, 0):
#         X, labels = data
#         X = X.to(device)
#         labels = labels.to(device)
#         outputs = deepmodel_MINST_3(X)
#         loss = loss_func(outputs, labels)
#         cur_loss += loss.item()
#         num_samples += 1

#         _, predictions = torch.max(outputs.data, 1)
#         total_count += labels.size(0)
#         correct_count += (predictions == labels).sum().item()
#     test_loss = cur_loss / num_samples
#     test_acc = 100 * correct_count / total_count

#     print("train loss and acc: ", test_loss, test_acc)

#     all_test_loss.append(test_loss)
#     all_test_acc.append(test_acc)

# plt.plot(all_alpha, all_train_loss, 'r*', label='train')
# plt.plot(all_alpha, all_test_loss, 'g^', label='test')
# plt.title("loss of Merged model")
# plt.xlabel("ratio")
# plt.ylabel("loss")
# plt.legend(loc="upper right")
# plt.savefig('HW-3-4')
# plt.close()

# plt.plot(all_alpha, all_train_acc, 'r*',  label='train')
# plt.plot(all_alpha, all_test_acc, 'g^', label='test')
# plt.title("accuracy of Merged model")
# plt.xlabel("ratio")
# plt.ylabel("accuracy")
# plt.legend(loc="lower right")
# plt.savefig('HW-3-5')


training_bs = [64, 96, 128, 256, 512]

all_train_loss = []
all_train_acc = []
all_test_loss = []
all_test_acc = []
all_train_sensitivity = []
all_test_sensitivity = []
all_batchsize = []
for batch_size in training_bs:
    deepmodel_MINST = DeepDNN(Ten_layer_params[1])
    deepmodel_MINST.to(device)
    optimizer1 = torch.optim.SGD(deepmodel_MINST.parameters(), lr=learning_rate)
    train_loss, train_acc, test_loss, test_acc, train_sensitivity, test_sensitivity = \
        training_process(deepmodel_MINST, optimizer1, num_epochs, batch_size)
    all_train_loss.append(train_loss)
    all_train_acc.append(train_acc)
    all_test_loss.append(test_loss)
    all_test_acc.append(test_acc)
    all_train_sensitivity.append(train_sensitivity)
    all_test_sensitivity.append(test_sensitivity)
    all_batchsize.append(batch_size)

plt.plot(all_batchsize, all_train_loss, 'r*', label='train')
plt.plot(all_batchsize, all_test_loss, 'g^', label='test')
plt.title("loss with different batch size")
plt.xlabel("batchsize")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig('HW-3-6')
plt.close()

plt.plot(all_batchsize, all_train_acc, 'r*', label='train')
plt.plot(all_batchsize, all_test_acc, 'g^', label='test')
plt.title("accuracy with different batch size")
plt.xlabel("batchsize")
plt.ylabel("accuracy")
plt.legend(loc="upper right")
plt.savefig('HW-3-7')
plt.close()

plt.plot(all_batchsize, all_train_sensitivity, 'r*', label='train')
plt.plot(all_batchsize, all_test_sensitivity, 'g^', label='test')
plt.title("Sensitivity with different batch size")
plt.xlabel("batchsize")
plt.ylabel("sensitivity")
plt.legend(loc="upper right")
plt.savefig('HW-3-8')
plt.close()