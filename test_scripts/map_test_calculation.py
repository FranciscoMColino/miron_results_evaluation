import numpy as np

# Provided data
thresholds = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
precision = np.array([1.00, 1.00, 0.61, 0.50, 0.50, 0.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
recall = np.array([1.00, 1.00, 0.62, 0.50, 0.50, 0.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

# Calculate Average Precision (AP)
ap = 0
for i in range(1, len(thresholds)):
    if recall[i] != recall[i-1]:  # Change in recall
        ap += precision[i] * (recall[i] - recall[i-1])

# Output the AP value
print(ap)