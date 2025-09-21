import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from fgsm import Attack

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_model(model, loader, device, attack=None, epsilon=0):
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        if attack and epsilon > 0:
            data = attack.generate_adversarial(data, target, nn.CrossEntropyLoss())
        
        output = model(data)
        pred = output.max(1)[1]
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load data
    train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
    
    # Quick training
    model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Quick training...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 100:  # Just 100 batches
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/100")
    
    model.eval()
    attack = Attack(model)
    
    print("MNIST FGSM Attack Test")
    print("-" * 25)
    
    clean_acc = test_model(model, test_loader, device)
    print(f"Accuracy: {clean_acc:.1f}%")
    
    for eps in [0.1, 0.5, 0.9]:
        adv_acc = test_model(model, test_loader, device, attack, eps)
        drop = clean_acc - adv_acc
        print(f"ε={eps}: {adv_acc:.1f}% (drop: {drop:.1f}%)")

if __name__ == "__main__":
    main()