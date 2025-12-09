import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_DIR = "output_directory"
BATCH_SIZE = 32
NUM_EPOCHS = 5

NUM_CLASSES = 6     #fist,no_gesture,peace,peace_inverted,stop,stop_inverted
DEVICE = torch.device("cuda")#use 3060ti GPU (for bobby)

#here, we use the same imagenet normalization as its original training
data_transforms = {
    "train": transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),]),
    "val": transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),]),
    "test": transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),]),}

image_datasets = {x:datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ["train", "val", "test"]}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == "train"))for x in ["train", "val", "test"]}

#load resnet
model = models.resnet18(pretrained=True)

#here, we replace final layer of resnet with our subset of classes from HaGRID
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=.001)#adjust learning rate

#now we train
for epoch in range(NUM_EPOCHS):
    print("Epoch:", epoch+1)
    for phase in ["train", "val"]:

        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                tst, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    #backward pass
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)
        epoch_loss = running_loss/len(image_datasets[phase])

        epoch_acc = running_corrects.float()/len(image_datasets[phase])


        print(f"On phase #{phase}, loss is {epoch_loss:.2f} and accuracy is{epoch_acc:.2f}")

#get test accuracy
model.eval()
correct = 0
total = 0



with torch.no_grad():
    for inputs, labels in dataloaders["test"]:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        tstt, preds = torch.max(outputs, 1)
        correct += torch.sum(preds==labels.data)
        total+=labels.size(0)
print("Test Accuracy is:", correct/total)



#save the final model
torch.save(model.state_dict(), "trained_gesture_resnet.pth")
