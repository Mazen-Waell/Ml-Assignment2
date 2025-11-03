def filter_binary(X, y, digit1=0, digit2=1):
    mask = (y == digit1) | (y == digit2)
    X_binary = X[mask]
    y_binary = y[mask]
    y_binary = (y_binary == digit2).astype(np.float32)
    return X_binary, y_binary

X_train_bin, y_train_bin = filter_binary(X_train, y_train)
X_val_bin, y_val_bin = filter_binary(X_val, y_val)
X_test_bin, y_test_bin = filter_binary(X_test, y_test)

X_train_t = torch.tensor(X_train_bin, dtype=torch.float32)
y_train_t = torch.tensor(y_train_bin, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val_bin, dtype=torch.float32)
y_val_t = torch.tensor(y_val_bin, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_bin, dtype=torch.float32)
y_test_t = torch.tensor(y_test_bin, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=128, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=128, shuffle=False)







class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(1))
    def forward(self, X):
        return torch.sigmoid(X @ self.weights + self.bias)

    def binary_cross_entropy(y_pred, y_true):
        eps = 1e-7
        return -torch.mean(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))

    def accuracy(y_pred, y_true):
        preds = (y_pred >= 0.5).float()
        return (preds == y_true).float().mean()


def binary_cross_entropy(y_pred, y_true):
    eps = 1e-7
    return -torch.mean(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))

def accuracy(y_pred, y_true):
    preds = (y_pred >= 0.5).float()
    return (preds == y_true).float().mean()





input_dim = X_train_t.shape[1]
model = LogisticRegressionModel(input_dim)
lr = 0.01
epochs = 20

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    model.train()
    total_loss, total_acc = 0, 0
    for Xb, yb in train_loader:
        y_pred = model(Xb)
        loss = binary_cross_entropy(y_pred, yb)
        loss.backward()
        with torch.no_grad():
            model.weights -= lr * model.weights.grad
            model.bias -= lr * model.bias.grad
        model.weights.grad.zero_()
        model.bias.grad.zero_()
        total_loss += loss.item()
        total_acc += accuracy(y_pred, yb).item()
    train_losses.append(total_loss / len(train_loader))
    train_accs.append(total_acc / len(train_loader))
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = 0, 0
        for Xb, yb in val_loader:
            y_pred = model(Xb)
            val_loss += binary_cross_entropy(y_pred, yb).item()
            val_acc += accuracy(y_pred, yb).item()
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc / len(val_loader))
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]*100:.2f}%")

