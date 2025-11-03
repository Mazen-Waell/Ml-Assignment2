class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        return logits


model_softmax = SoftmaxRegression(input_dim=784, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_softmax.parameters(), lr=0.01)


def train_softmax(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.squeeze().long())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch.squeeze()).sum().item()
            total += y_batch.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.squeeze().long())
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch.squeeze()).sum().item()
                val_total += y_batch.size(0)

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs




train_losses, val_losses, train_accs, val_accs = train_softmax(
    model_softmax, train_loader, val_loader, criterion, optimizer, epochs=20
)





