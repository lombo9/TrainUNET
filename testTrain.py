import torch
import torch.optim as optim
from torch.utils.data import random_split
from testModel import UNETImproved
from testDataset import get_isic_dataloader
from torchvision.utils import save_image
import os

output_dir = "/home/Student/s4585713/TrainUNET"
os.makedirs(output_dir, exist_ok=True)

num_epochs = 3
batch_size = 128
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/Student/s4585713/TrainUNET/ISIC-2017_Training_Data/ISIC-2017_Training_Data"
full_loader = get_isic_dataloader(root_dir, batch_size=batch_size)
train_size = int(0.8 * len(full_loader.dataset))
val_size = len(full_loader.dataset) - train_size
train_dataset, val_dataset = random_split(full_loader.dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = UNETImproved(n_classes=2).to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def dice_coefficient(prediction, target):
    num = prediction.size(0)
    x = prediction.view(num, -1).float()
    y = target.view(num, -1).float()
    intersect = (x * y).sum().float()
    return (2 * intersect) / (x.sum() + y.sum())

train_loss_history = []
train_dice_history = []
val_loss_history = []
val_dice_history = []

train_dice_total = 0.0
val_dice_total = 0.0
for epoch in range(num_epochs):
    print(f"\n==== Epoch {epoch} ====")

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            print(f"\n---- Training ----")

            train_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                dice_val = dice_coefficient(outputs, labels)
                train_dice_total += dice_val.item()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                print(f"Batch {i}/{len(train_loader)}", end="\r")

            print(f"\n\t Average Training Loss: {train_loss / len(train_loader)}")
            print(f"\n\t Average Training Dice Coefficient: {train_dice_total / len(train_loader)}")

            train_loss_history.append(train_loss / len(train_loader))
            train_dice_history.append(train_dice_total / len(train_loader))

        elif phase == 'val':
            model.eval()
            print(f"\n---- Validation ----")

            val_loss = 0.0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    dice_val = dice_coefficient(outputs, labels)
                    val_dice_total += dice_val.item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    print(f"Batch {i}/{len(val_loader)}", end="\r")

            print(f"\n\t Average Validation Loss: {val_loss / len(val_loader)}")
            print(f"\n\t Average Validation Dice Coefficient: {val_dice_total / len(val_loader)}")

            val_loss_history.append(val_loss / len(val_loader))
            val_dice_history.append(val_dice_total / len(val_loader))

            samples = 3
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_loader):
                    if i == samples:
                        break
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    save_image(outputs[0].cpu(), os.path.join(output_dir, f"sample_output_{epoch}_{i}.png"))
                    save_image(labels[0].cpu(), os.path.join(output_dir, f"sample_label_{epoch}_{i}.png"))
                    save_image(inputs[0].cpu(), os.path.join(output_dir, f"sample_input_{epoch}_{i}.png"))


print("\nTrain Loss History:", train_loss_history)
print("\nTrain Dice Coefficient History:", train_dice_history)
print("\nValidation Loss History:", val_loss_history)
print("\nValidation Dice Coefficient History:", val_dice_history)
