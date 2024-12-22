import torch
from data.data_loader import load_data
from models.capsule_network import CapsuleNetwork

def train_model():
    dataloader = load_data()
    model = CapsuleNetwork(input_shape=(3, 64, 64), num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        for batch in dataloader:
            rgb, hsi, sonar, labels = batch['rgb'], batch['hsi'], batch['sonar'], batch['label']
            outputs = model(rgb)  # Example: Modify to include multimodal fusion
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    train_model()
