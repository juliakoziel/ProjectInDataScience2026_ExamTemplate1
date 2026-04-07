#will load dataset and split into train, validation, test

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from skimage.io import imread
import matplotlib.pyplot as plt


ff = pd.read_csv(r"C:\Users\user\Documents\ProjectInDataScience2026_Exercises\data\metadata_with_group.csv")
df = ff[ (ff["group_id"]== "G") | (ff["group_id"]=="K") | (ff["group_id"]=="E") ].copy()
cancerous_diagnostics = ['BCC', 'MEL', 'SCC'] 
df['cancer'] = df['diagnostic'].isin(cancerous_diagnostics).astype(int) #1 if the diagnostic is in the list, and 0 otherwise
df.head(5)


df["path"] = "E:\projects\ProjectInDataScience2026_ExamTemplate1\data\imgs\\" + df["img_id"] 

X = df["path"].values
y = df["cancer"].values

X_train, X_teva, y_train, y_teva = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_teva, y_teva,
    test_size=0.50,
    stratify=y_teva,
    random_state=42
)


#now to see if it worked###############
#load one img
img = imread(df["path"].iloc[0])

plt.imshow(img)
plt.title("Test image")
plt.axis("off")
plt.show()

