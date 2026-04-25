from skimage.io import imread
from skimage.transform import resize
from skimage import morphology
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from split_data_in_3sets import X_test, X_train

#1 Load dataset and split into train, validation, test

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from skimage.io import imread
import matplotlib.pyplot as plt


#first we add a new colloumn
ff = pd.read_csv(r"E:\Documents\ProjectInDataScience2026_Exercises\data\metadata_with_group.csv")
df = ff[ (ff["group_id"]== "G") | (ff["group_id"]=="K") | (ff["group_id"]=="E") ].copy()
cancerous_diagnostics = ['BCC', 'MEL', 'SCC'] 
df['cancer'] = df['diagnostic'].isin(cancerous_diagnostics).astype(int) #1 if the diagnostic is in the list, and 0 otherwise
df.head(5)


#now we split into sets
#to create image paths (adjust folder if needed)
df["path"] = "E:\projects\ProjectInDataScience2026_ExamTemplate1\data\imgs\\" + df["img_id"] 

# Extract X and y
X = df["path"].values
y = df["cancer"].values

# Split
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


#now to see if it workedd########################################

#load one img
img = imread(df["path"].iloc[9])

plt.imshow(img)
plt.title("Test image")
plt.axis("off")
plt.show()

#to check if paths exist
for i in range(3):
    print(df["path"].iloc[i])
    print(os.path.exists(df["path"].iloc[i]))


#checks how big the sets are
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))



from skimage.io import imread
from skimage.transform import resize
from skimage import morphology
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from split_data_in_3sets import X_test, X_train
 
 
def preprocess_img(path, size=(224, 224)):
 
    def load_img(p):
        return imread(p)
 
    def remove_artifacts(img):
        img_uint8 = img.copy()
 
        if img_uint8.dtype in (np.float32, np.float64):
            img_uint8 = (np.clip(img_uint8, 0, 1) * 255).astype(np.uint8)
 
        if img_uint8.ndim == 2:
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        elif img_uint8.shape[2] == 4:
            img_uint8 = img_uint8[:, :, :3]
 
        # ── 1. HAIR DETECTION ────────────────────────────────────────────────
        #
        # Fix for over-detection: bilateral blur before black-hat suppresses
        # skin texture. Otsu threshold on the black-hat response self-calibrates
        # per image. Blob aspect-ratio filter keeps only elongated (hair-shaped)
        # regions and rejects large blobby skin patches.
        #
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        gray_smooth = cv2.bilateralFilter(gray, 9, 30, 30)
 
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blackhat = cv2.morphologyEx(gray_smooth, cv2.MORPH_BLACKHAT, kernel)
 
        otsu_val, _ = cv2.threshold(blackhat, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hair_thresh = max(10, int(otsu_val * 0.6))
        _, hair_mask = cv2.threshold(blackhat, hair_thresh, 255, cv2.THRESH_BINARY)
 
        disk2 = morphology.disk(2)
        disk3 = morphology.disk(3)
        hair_mask = morphology.binary_opening(hair_mask > 0, disk2)
        hair_mask = morphology.binary_closing(hair_mask,    disk3)
        hair_mask = (hair_mask * 255).astype(np.uint8)
 
        # Keep only elongated blobs → hair. Skin texture = large, round → rejected.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask, 8)
        filtered_hair = np.zeros_like(hair_mask)
        img_area = hair_mask.size
        for lbl in range(1, n_labels):
            area   = stats[lbl, cv2.CC_STAT_AREA]
            w      = stats[lbl, cv2.CC_STAT_WIDTH]
            h      = stats[lbl, cv2.CC_STAT_HEIGHT]
            aspect = max(w, h) / max(min(w, h), 1)
            if area < img_area * 0.02 and aspect > 2.5:
                filtered_hair[labels == lbl] = 255
        hair_mask = filtered_hair
 
        hair_mask = morphology.binary_dilation(hair_mask > 0, disk2)
        hair_mask = (hair_mask * 255).astype(np.uint8)
 
        # ── 2. BLUE / PURPLE-BLUE PEN DETECTION ──────────────────────────────
        #
        # Surgical/dermoscopy markers are a desaturated purple-blue.
        # They appear in HSV with hue 105-170 and often low saturation.
        # LAB b* < 123 (shifted blue) confirms the colour and cuts false
        # positives from pink/red lesion areas.
        #
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
 
        pen_hsv = cv2.inRange(hsv,
                              np.array([105,  10,  20]),
                              np.array([170, 220, 220]))
 
        b_star  = lab[:, :, 2].astype(np.int16)
        a_star  = lab[:, :, 1].astype(np.int16)
        lab_pen = ((b_star < 123) & (a_star < 138)).astype(np.uint8) * 255
 
        blue_mask = cv2.bitwise_and(pen_hsv, lab_pen)
 
        # Exclude very dark pixels (shadows, not ink).
        blue_mask[hsv[:, :, 2] < 20] = 0
 
        disk1 = morphology.disk(1)
        disk5 = morphology.disk(5)
        blue_mask = morphology.binary_opening(blue_mask > 0, disk1)
        blue_mask = morphology.binary_closing(blue_mask,    disk5)
        blue_mask = morphology.binary_dilation(blue_mask,   disk3)
        blue_mask = (blue_mask * 255).astype(np.uint8)
 
        # Drop isolated specks < 8 px.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, 8)
        filtered_blue = np.zeros_like(blue_mask)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= 8:
                filtered_blue[labels == lbl] = 255
        blue_mask = filtered_blue
 
        # ── 3. MERGE & INPAINT ───────────────────────────────────────────────
        combined_mask = cv2.bitwise_or(hair_mask, blue_mask)
 
        inpaint_r = max(3, int(min(img_uint8.shape[:2]) * 0.01))
        cleaned   = cv2.inpaint(img_uint8, combined_mask, inpaint_r,
                                cv2.INPAINT_TELEA)
        cleaned   = cleaned.astype(np.float32) / 255.0
 
        return cleaned, combined_mask
 
    def resize_image(img):
        return resize(img, size, anti_aliasing=True)
 
    img = load_img(path)
    img_cleaned_float, combined_mask = remove_artifacts(img)
    img_resized = resize_image(img_cleaned_float)
 
    return img_resized, combined_mask
 
 
# ── Visual sanity check ───────────────────────────────────────────────────────
img_clean, combined_mask = preprocess_img(X_train[12])
original = imread(X_train[12])
 
plt.figure(figsize=(12, 4))
 
plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original")
plt.axis('off')
 
plt.subplot(1, 3, 2)
plt.imshow(combined_mask, cmap='gray')
plt.title("Detected Artifacts")
plt.axis('off')
 
plt.subplot(1, 3, 3)
plt.imshow(img_clean)
plt.title("Cleaned Image")
plt.axis('off')
 
plt.tight_layout()
plt.show()