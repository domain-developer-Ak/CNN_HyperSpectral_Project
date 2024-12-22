import torch
from torch.utils.data import DataLoader
from data_loader import MultiModalDataset, load_data
from models.capsule_network import CapsuleNetwork
from utils.metrics import calculate_accuracy, calculate_confusion_matrix

def evaluate_model(model_path, batch_size=32):
    # Load test data
    test_loader = load_data(batch_size=batch_size)

    # Load the trained model
    model = CapsuleNetwork(input_shape=(3, 64, 64), num_classes=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            rgb, hsi, sonar, labels = batch['rgb'], batch['hsi'], batch['sonar'], batch['label']
            outputs = model(rgb)  # Example: Modify to include multimodal fusion
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = calculate_accuracy(all_preds, all_labels)
    confusion_matrix = calculate_confusion_matrix(all_preds, all_labels)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix)

if __name__ == "__main__":
    evaluate_model(model_path="saved_model.pth")
