import os
import sys
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from datasets.PcamDataset import PCamDataset
from model_files.ResnetPcam import create_resnet18_for_pcam

'''
Original code used to create the trained model.
Model is created with resnet-18 architecture and trained with the full train dataset.

'''
if __name__ == "__main__":

    #path hidden for privacy, in future use, this can be improved to be dynamic path and not hard coded
    train_x_path = r"\camelyonpatch_level_2_split_train_x.h5"
    train_y_path = r"\camelyonpatch_level_2_split_train_y.h5"
    metadata_path = r"\camelyonpatch_level_2_split_test_meta.csv"
    model_save_dir = r"\trained-models"
    os.makedirs(model_save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((96, 96), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    #caclulate number of whole slide images contained into the train dataset
    print("Loading metadata...")
    metadata = pd.read_csv(metadata_path)
    unique_wsis = metadata["wsi"].unique()
    print(f"Found {len(unique_wsis)} unique whole slide images (WSI) in metadata.")

    #split the train dataset into two parts, 80% used for train and 20% for validate
    train_wsis, val_wsis = train_test_split(
        unique_wsis, test_size=0.2, random_state=42
    )
    train_indices = metadata[metadata["wsi"].isin(train_wsis)].index.tolist()
    val_indices = metadata[metadata["wsi"].isin(val_wsis)].index.tolist()
    print(f"Image level split: {len(train_wsis)} train WSIs, {len(val_wsis)} val WSIs.")

    print("Loading dataset from HDF5...")
    full_dataset = PCamDataset(train_x_path, train_y_path, transform=transform)
    print(f"Loaded dataset with {len(full_dataset)} samples.")
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"Final datasets: {len(train_dataset)} train, {len(val_dataset)} validation samples.")

    #apply the train labels 
    train_labels = [metadata.iloc[idx]["tumor_patch"] for idx in train_indices]

    #Tumor = 1, Normal = 0
    train_labels = [1 if lbl else 0 for lbl in train_labels]
    label_counts = Counter(train_labels)
    print(f"Class counts in train set: {label_counts}")
    class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cpu")

    #call method to create model from ResnetPcam.py
    model = create_resnet18_for_pcam().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    weights_tensor = torch.tensor([1.0, 2.0]).to(device)
    criterion = CrossEntropyLoss(weight=weights_tensor)
    best_val_acc = 0.0

    #initiate train in epochs, with total epochs = 10. can modify for more refined understanding of data 
    for epoch in range(1, 11):
        print(f"\nEpoch {epoch} running...")
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        #for each batch calculate loss, total predictions and correct predictions
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} processed")

        #for each epoch calculate and print out its train accuracy and loss
        train_acc = 100 * correct / total
        print(f"Epoch {epoch} Train Summary: Loss={total_loss:.3f}, Accuracy={train_acc:.2f}%")

        model.eval()
        val_correct, val_total = 0, 0
        class_correct = [0, 0]
        class_total = [0, 0]

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                for label, pred in zip(labels, predicted):
                    class_total[label.item()] += 1
                    if label.item() == pred.item():
                        class_correct[label.item()] += 1
        
        #for each epoch calculate and print out its validation accuracy and loss
        val_acc = 100 * val_correct / val_total
        tumor_recall = (100 * class_correct[1] / class_total[1]) if class_total[1] else 0.0
        print(f"Validation Accuracy: {val_acc:.2f}%, Tumor Recall: {tumor_recall:.2f}%")

        #if outcome of epoch is better from previous, save this epoch's model as best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(model_save_dir, f"resnet18_pcam_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved to {best_path} with val acc {val_acc:.2f}%")

        #save check point after epoch finish for version control
        save_path = os.path.join(model_save_dir, f"resnet18_pcam_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Epoch checkpoint saved to {save_path}")

    final_path = os.path.join(model_save_dir, "resnet18_pcam_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completed succesfully. Final model saved to {final_path}")
