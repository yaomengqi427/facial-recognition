import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold

def load_data(location):
    list1 = os.listdir(location)
    list_title = []
    list_pho = []
    for i in range(0, 40):
        path1 = os.path.join(location, list1[i])
        list2 = os.listdir(path1)
        for k in range(0, len(list2)):
            path2 = os.path.join(path1, list2[k])
            raw_image = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            list_pho.append(list(raw_image.flatten()))
            list_title.append(i + 1)
    return list_pho, list_title

def compute_lbp(binary_image, radius=3, n_points=8):
    lbp = local_binary_pattern(binary_image, n_points, radius, method='uniform')
    return lbp

def compute_histogram(lbp_matrix, num_bins=256):
    histogram = np.zeros(num_bins)
    for i in range(lbp_matrix.shape[0]):
        for j in range(lbp_matrix.shape[1]):
            histogram[int(lbp_matrix[i, j])] += 1
    return histogram / np.sum(histogram)

def extract_features(input_image):
    binary_image = input_image.convert('1')
    lbp_matrix = compute_lbp(binary_image)
    histogram = compute_histogram(lbp_matrix)
    return histogram

def pca_reduction(list_pho, n_components=150):
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(list_pho)
    list_pho_pca = pca.transform(list_pho)
    return list_pho_pca

def svm_classification_with_cv(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = svm.SVC(C=1)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)
    average_accuracy = np.mean(accuracies)
    print(f"每次交叉验证的准确率: {accuracies}")
    print(f"平均准确率: {average_accuracy:.4f}")
    return average_accuracy

def visualize_lbp_features(images, lbp_features, histograms):
    for i in range(len(images)):
        plt.figure(figsize=(10, 4))
        plt.suptitle(f'第 {i + 1}张图片', fontsize=12)
        plt.subplot(131)
        plt.imshow(images[i], cmap='gray')
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(lbp_features[i], cmap='gray')
        plt.title('LBP特征图像')
        plt.axis('off')

        plt.subplot(133)
        plt.bar(range(len(histograms[i])), histograms[i])
        plt.title('LBP特征直方图')
        plt.xlabel('区间')
        plt.ylabel('频率')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)
        plt.close()

def main():
    location = "./orl"
    list_pho, list_title = load_data(location)
    images = []
    lbp_features = []
    histograms = []
    plt.rcParams['font.sans-serif'] = ['SimHei']

    for i in range(len(list_pho)):
        image = np.array(list_pho[i]).reshape(112, 92)
        lbp_image = local_binary_pattern(image, 8, 1, method='uniform')
        hist = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 8))[0]
        images.append(image)
        lbp_features.append(lbp_image)
        histograms.append(hist)

    visualize_lbp_features(images, lbp_features, histograms)

    list_pho_pca = pca_reduction(list_pho)
    list_pho_pca = np.array(list_pho_pca)
    list_title = np.array(list_title)

    average_accuracy = svm_classification_with_cv(list_pho_pca, list_title)

if __name__ == "__main__":
    main()