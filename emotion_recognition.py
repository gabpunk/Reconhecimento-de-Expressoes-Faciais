from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, classification_report


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertendo para escala de cinza
    transforms.Resize((48, 48)),                  # Redimensionar para 48x48 pixels
    transforms.ToTensor(),                        # Converter para tensor
    transforms.Normalize((0.5,), (0.5,))          # Normalizar
])

train_dataset = datasets.ImageFolder('Facial_Emotion_Dataset/train', transform=transform)
test_dataset = datasets.ImageFolder('Facial_Emotion_Dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

emotion_classes = train_dataset.classes
print("Classes de Emoções:", emotion_classes)


class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1_input_dim = None  
        self.fc1 = None
        self.fc2 = nn.Linear(128, len(emotion_classes)) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        
        if self.fc1_input_dim is None:
            self.fc1_input_dim = x.numel() // x.size(0)
            self.fc1 = nn.Linear(self.fc1_input_dim, 128).to(x.device)  # Inicializa fc1 dinamicamente

        # Flatten
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


num_epochs = 5
for epoch in range(num_epochs):
    print(f"Iniciando época {epoch + 1}")
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

all_labels = []
all_predictions = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

print(confusion_matrix(all_labels, all_predictions))
print(classification_report(all_labels, all_predictions, target_names=emotion_classes))

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Acurácia nos dados de teste: {100 * correct / total:.2f}%')


torch.save(model.state_dict(), 'model_emotion.pth')

# Para carregar posteriormente:
# model.load_state_dict(torch.load('model_emotion.pth'))
# model.eval()
