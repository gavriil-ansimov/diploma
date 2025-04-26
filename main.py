
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import os
from PIL import Image
from sklearn.model_selection  import train_test_split
from ManiFeSt import ManiFeSt
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

num_of_samples = 300
def load_and_resize_images(folder_path, target_size=(64,64)):
    images = []
    files = os.listdir(folder_path)
    indx = random.sample(range(len(files)), 150)
    files = np.array([files[i] for i in indx])

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        #img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = Image.open(filepath)

        r, g, b = img.split()

        ra = np.array(r)
        ga = np.array(g)
        ba = np.array(b)

        gray = (0.299 * ra + 0.587 * ga + 0.114 * ba)

        img_resized = cv2.resize(gray, target_size)
        fourier_transform = np.fft.fft2(img_resized)

        # Shift the zero frequency component to the center
        ft_shifted = np.fft.fftshift(fourier_transform)
        m_spectrum = np.abs(ft_shifted)
        img = np.array(img_resized)

        images.append(img)
    return images

#target_size = (new_width, new_height)  # Define the target size for resizing

cancer_folder = "..."
non_cancer_folder = "..."

cancer_images = load_and_resize_images(cancer_folder)
print(len(cancer_images))
non_cancer_images = load_and_resize_images(non_cancer_folder)
print(len(non_cancer_images))

X_cancer = np.array(cancer_images)
X_non_cancer = np.array(non_cancer_images)

y_cancer = np.ones(len(X_cancer))  # Label 1 for cancer
y_non_cancer = np.zeros(len(X_non_cancer))  # Label 0 for non-cancer

y = np.concatenate((y_cancer, y_non_cancer))
X = np.concatenate((X_cancer, X_non_cancer))
X = X.reshape(X.shape[0],-1)
print(X[0])
print(X[0].shape)

shuffle_indices = np.random.permutation(len(X))
X = X[shuffle_indices]
y = y[shuffle_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y, random_state=40)

score, idx, eig_vecs = ManiFeSt(X_train,y_train,kernel_scale_factor=True,use_spsd=True)  #use_spsd=use_spsd
print(score.shape, idx.shape)
# %% Plot Score
label = list(set(y_train))
x_train_1 = X_train[np.where(y_train == 1)]
x_train_0 = X_train[np.where(y_train == 0)]
(eigVecD, eigValD, eigVecM, eigValM) = eig_vecs

plt.rc('text', usetex=True)

fig = plt.figure(figsize=(6.75, 3.53), constrained_layout=False, facecolor='0.9', dpi=500)
gs = fig.add_gridspec(nrows=22, ncols=42, left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

# plot samples from each class
ax = fig.add_subplot(gs[1:6, 0:5])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4,
                                              subplot_spec=gs[1:6, 0:5], wspace=0.0, hspace=0.0)
for j in range(16):
    ax = plt.Subplot(fig, inner_grid[j])
    im = ax.imshow(abs(x_train_0[j, :].reshape((64, 64))), cmap=plt.get_cmap('gray'))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

ax = fig.add_subplot(gs[17:22, 0:5])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4,
                                              subplot_spec=gs[17:22, 0:5], wspace=0.0, hspace=0.0)
for j in range(16):
    ax = plt.Subplot(fig, inner_grid[j])
    im = ax.imshow(abs(x_train_1[j, :].reshape((64, 64))), cmap=plt.get_cmap('gray'))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

# plot eigenvectors Mean operator M
ax = fig.add_subplot(gs[7:11, 0:4])
im = ax.imshow(abs(eigVecM[:, 0].reshape((64, 64))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[12:16, 0:4])
im = ax.imshow(abs(eigVecM[:, 1].reshape((64, 64))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

# plot eigenvectors of Difference operator D
ax = fig.add_subplot(gs[2:6, 18:22])
im = ax.imshow(abs(eigVecD[:, 0].reshape((64, 64))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[7:11, 18:22])
im = ax.imshow(abs(eigVecD[:, 1].reshape((64, 64))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[12:16, 18:22])
im = ax.imshow(abs(eigVecD[:, 2].reshape((64, 64))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[17:21, 18:22])
im = ax.imshow(abs(eigVecD[:, 3].reshape((64, 64))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

# plot ManiFeSt Score
ax = fig.add_subplot(gs[7:16, 33:42])
im = ax.imshow(abs(score.reshape((64, 64))))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
plt.show()

num_of_features = [500, 700, 800]
num_cv_folds = 10  # Number of cross-validation folds

for n in num_of_features:
    feature_subset = np.zeros((X_train.shape[0], n))
    for i in range(X_train.shape[0]):
        feature_subset[i] = X_train[i, idx[:n]]  # Select the top 'n' features using sorted indices 'idx'

    # Initialize StratifiedKFold for stratified cross-validation
    skf = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=42)

    # Initialize list to store cross-validation scores
    cv_scores = []

    # Iterate over cross-validation folds
    for train_index, test_index in skf.split(feature_subset, y_train):
        X_train_cv, X_test_cv = feature_subset[train_index], feature_subset[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

        # Train SVM Classifier
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(X_train_cv, y_train_cv)

        # Evaluate Performance on the test fold
        y_pred_cv = svm_classifier.predict(X_test_cv)
        accuracy_cv = accuracy_score(y_test_cv, y_pred_cv)
        cv_scores.append(accuracy_cv)

    # Calculate and print average cross-validation score
    avg_cv_score = np.mean(cv_scores)
    print("Num of features:", n)
    print("Average Cross-Validation Accuracy:", avg_cv_score)











