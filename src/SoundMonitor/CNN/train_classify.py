from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# X_train: List of 1024-D embedding vectors extracted from your audio files
# y_train: Labels (1 = Compressor ON, 0 = Compressor OFF)

classifier = SVC(kernel="rbf", C=1.0)
classifier.fit(X_train, y_train)

# Fast inference for real-time streaming
y_pred = classifier.predict(X_test)
print(f"Accuracy with pre-trained embeddings: {accuracy_score(y_test, y_pred):.2f}")
