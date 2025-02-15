"""
Execution output of 68 videos of 10-to-30sec of movie trailers: 
===============================================================

Results of the Analysis:
-----------------------

Accuracy: 0.632 (63.2%)

Confusion Matrix:
[[ 4  3  8]  # True Column 1
 [ 0  5 13]  # True Column 2
 [ 0  1 34]] # True Column 3

Detailed Analysis:
Total samples: 68
Correct predictions: 43
Incorrect predictions: 25

Per-Class Metrics:
Column 1:
- Accuracy: 0.838 (83.8%)
- True instances: 15
- Predicted instances: 4

Column 2:
- Accuracy: 0.750 (75.0%)
- True instances: 18
- Predicted instances: 9

Column 3:
- Accuracy: 0.676 (67.6%)
- True instances: 35
- Predicted instances: 55
"""

from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# True values
y_true = [
    [0,1,0], [1,0,0], [0,0,1], [0,0,1], [0,0,1], [0,0,1],
    [0,1,0], [0,0,1], [1,0,0], [1,0,0], [0,1,0], [0,1,0],
    [0,1,0], [1,0,0], [0,0,1], [0,0,1], [0,0,1], [0,0,1],
    [0,1,0], [0,0,1], [0,0,1], [1,0,0], [0,0,1], [0,1,0],
    [0,0,1], [0,1,0], [1,0,0], [0,0,1], [1,0,0], [1,0,0],
    [0,1,0], [0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,0,1],
    [0,1,0], [0,0,1], [0,0,1], [0,1,0], [0,0,1], [0,0,1],
    [0,0,1], [1,0,0], [0,1,0], [0,0,1], [0,1,0], [0,0,1],
    [0,0,1], [1,0,0], [1,0,0], [0,1,0], [0,0,1], [0,0,1],
    [0,0,1], [0,0,1], [1,0,0], [0,1,0], [0,0,1], [0,1,0],
    [0,0,1], [0,0,1], [0,0,1], [1,0,0], [1,0,0], [1,0,0],
    [0,1,0], [0,0,1]
]

# Predicted values
y_pred = [
    [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1],
    [0,0,1], [0,0,1], [1,0,0], [0,1,0], [0,0,1], [0,0,1],
    [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1],
    [0,1,0], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1],
    [0,0,1], [0,0,1], [0,0,1], [0,0,1], [1,0,0], [0,0,1],
    [0,1,0], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1],
    [0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,0,1], [0,0,1],
    [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1],
    [0,0,1], [0,1,0], [0,0,1], [0,0,1], [0,0,1], [0,1,0],
    [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,1,0],
    [0,0,1], [0,0,1], [0,0,1], [1,0,0], [0,1,0], [1,0,0],
    [0,1,0], [0,0,1]
]

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Get the class predictions (convert from one-hot to class indices)
y_true_class = np.argmax(y_true, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

# Calculate accuracy
acc = accuracy_score(y_true_class, y_pred_class)
print(f"Accuracy: {acc:.3f}")

# Calculate and display confusion matrix
cm = confusion_matrix(y_true_class, y_pred_class)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix using matplotlib instead of seaborn
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Add labels
class_labels = ['Column 1', 'Column 2', 'Column 3']
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)

# Add text annotations to the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Print detailed analysis
print("\nDetailed Analysis:")
print(f"Total samples: {len(y_true)}")
print(f"Correct predictions: {np.sum(y_true_class == y_pred_class)}")
print(f"Incorrect predictions: {np.sum(y_true_class != y_pred_class)}")

# Calculate per-class metrics
for i, class_name in enumerate(class_labels):
    true_class = y_true_class == i
    pred_class = y_pred_class == i
    class_acc = accuracy_score(true_class, pred_class)
    print(f"\n{class_name}:")
    print(f"Accuracy: {class_acc:.3f}")
    print(f"True instances: {np.sum(true_class)}")
    print(f"Predicted instances: {np.sum(pred_class)}")
