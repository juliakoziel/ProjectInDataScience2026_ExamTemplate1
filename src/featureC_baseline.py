#color complexity feature (Measures color variation: Uniform → benign, Many colors → melanoma)
import matplotlib.pyplot as plt
import numpy as np
from clean_the_imgs import preprocess_img

def color_complexity_B(path):
   
    img, _ = preprocess_img(path)  
    pixels = img.reshape(-1, 3)  
    std_per_channel = np.std(pixels, axis=0)  
    color_complexity = np.mean(std_per_channel)   
    return color_complexity
 
img, _ = preprocess_img(X_train[66])
pixels = img.reshape(-1, 3)
value = color_complexity_B(X_train[66])
print(value)
plt.figure(figsize=(10,4))
 
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Cleaned Image")
plt.axis('off')
 
plt.subplot(1,2,2)
plt.hist(pixels[:,0], bins=50, alpha=0.5, label='R')
plt.hist(pixels[:,1], bins=50, alpha=0.5, label='G')
plt.hist(pixels[:,2], bins=50, alpha=0.5, label='B')
plt.title("Color Distribution")
plt.legend()
plt.show()