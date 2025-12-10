import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = "/Users/jarytolentino/Documents/Deep Learning for CV/output"

    # https://medium.com/@RobuRishabh/understanding-and-implementing-mobilenetv3-422bd0bdfb5a#


    # creating variation in the training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    # defining transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # load datasets
    train_dataset = datasets.ImageFolder(root = f"{root}/train", transform = train_transform)
    test_dataset = datasets.ImageFolder(root = f"{root}/test", transform = transform)
    val_dataset = datasets.ImageFolder(root = f"{root}/val", transform = transform)

    num_classes = 6
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 4)

    # load the pretrained model
    mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)

    # fine-tune on our dataset
    mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features = num_classes)
    mobilenet_v3_large = mobilenet_v3_large.to(device)

    # Train the model
    criterion = nn.CrossEntropyLoss()

    # Parameter groups: small LR for backbone, larger for classifier
    optimizer = optim.Adam([
        {"params": mobilenet_v3_large.features.parameters(),   "lr": 1e-4},
        {"params": mobilenet_v3_large.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    count = 0

    # Training Loop
    num_epochs = 20
    best_val_loss = float('inf')
    best_model_path = 'mobilenetv3.pkl'

    for epoch in range(num_epochs):
        mobilenet_v3_large.train()
        train_loss = 0.0
        train_corrects = 0
        total = 0

        # for inputs, labels in train_loader:
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = mobilenet_v3_large(inputs)
            
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # get the label with the highest output
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = train_loss / total
        epoch_acc = train_corrects / total

        # Evaluate on validation set
        mobilenet_v3_large.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            # for inputs, labels in val_loader:
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = mobilenet_v3_large(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum().item()
                val_total += labels.size(0)
            
        val_loss = val_loss / val_total
        val_acc = val_corrects / val_total
        

        print(f"Epoch [{epoch + 1}/{num_epochs}]\n"
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}\n"
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

        # stop training when there arent improvements on validation set
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mobilenet_v3_large.state_dict(), best_model_path)
            #reset counter
            count  = 0
        
        elif val_loss > best_val_loss and best_val_loss != float('inf'):
            count += 1
            if count >= 5:
                print("stopping training")
                break


    # Evaluate on the test set
    mobilenet_v3_large.load_state_dict(torch.load(best_model_path))
    mobilenet_v3_large.eval()

    test_loss = 0.0
    test_corrects = 0
    test_total = 0
    with torch.no_grad():
        # for inputs, labels in test_loader:
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = mobilenet_v3_large(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += (preds == labels).sum().item()
            test_total += labels.size(0)
        
    test_loss = test_loss / test_total
    test_acc = test_corrects / test_total


    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()