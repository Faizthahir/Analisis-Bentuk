import cv2
from matplotlib.pyplot import contour
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

def load_dataset(path):
    images = []
    labels = []

    for label in os.listdir(path):
        folder = os.path.join(path, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is not None:
                images.append(img)
                labels.append(label)

    return images, labels

def get_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    return max(contours, key=cv2.contourArea)


def region_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0
    extent = area / (w * h) if (w*h) != 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    return [area, perimeter, aspect_ratio, extent, solidity]


def extract_features(images):
    features = []
    valid_idx = []

    for i, img in enumerate(images):
        contour = get_contour(img)

        if contour is None:
            print(f"[WARNING] Contour gagal pada gambar ke-{i}")
            continue

        f1 = region_features(contour)

        M = cv2.moments(contour)
        hu = cv2.HuMoments(M).flatten()
        f = f1 + list(hu)

        features.append(f)
        valid_idx.append(i)

    return np.array(features), valid_idx


def classify(X, y):
    print("\n=== TANPA SPLIT (CEK OVERFITTING) ===")
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)
    y_pred = model.predict(X)

    print("Akurasi (train=test):", accuracy_score(y, y_pred))

    print("\n=== TRAIN TEST SPLIT (STRATIFIED) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Label test:", y_test)
    print("Prediksi :", y_pred)
    print("Akurasi (split):", accuracy_score(y_test, y_pred))

    print("\n=== CROSS VALIDATION ===")
    scores = cross_val_score(model, X, y, cv=3)

    print("Akurasi per fold:", scores)
    print("Akurasi rata-rata:", scores.mean())


def main():
    dataset_path = "dataset"

    images, labels = load_dataset(dataset_path)

    print("Total gambar:", len(images))

    features, valid_idx = extract_features(images)

    labels = [labels[i] for i in valid_idx]

    print("Jumlah fitur:", len(features))
    print("Jumlah label:", len(labels))

    if len(features) == 0:
        print("Tidak ada fitur yang berhasil diekstrak!")
        return

    classify(features, labels)


if __name__ == "__main__":
    main()