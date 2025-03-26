import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Simulate medical image data (e.g., pixel values)
# Create 100 simulated images, each represented by 256 pixel values
X = np.random.rand(100, 256) 
# Simulate binary labels (0: healthy, 1: tumor)
y = np.random.randint(0, 2, 100)  

# Train a simple AI model using RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)  

# Predict on new data
new_image = np.random.rand(1, 256)  
prediction = model.predict(new_image)  

# Print the result based on the prediction
print("Tumor detected!" if prediction[0] == 1 else "No tumor detected.")