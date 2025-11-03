train_df = pd.read_csv('data/mnist_train.csv')
test_df = pd.read_csv('data/mnist_test.csv')

# Extract features and labels
X_train_full = train_df.iloc[:, 1:].values
y_train_full = train_df.iloc[:, 0].values
X_test_full  = test_df.iloc[:, 1:].values
y_test_full  = test_df.iloc[:, 0].values

# Normalize to [0,1]
X_train_full = X_train_full / 255.0
X_test_full  = X_test_full / 255.0

# Combine and split
X_all = np.concatenate([X_train_full, X_test_full], axis=0)
y_all = np.concatenate([y_train_full, y_test_full], axis=0)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.4, random_state=42, stratify=y_all
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")



