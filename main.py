# 노동요 - 띠따띠라띠따따뚜따(Tick-Tack by ILLIT)
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

ks = range(1, 42, 2)
digits = load_digits()
X, y = digits.data, digits.target

def evaluation(p):
    mean_scores = []

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=p)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        mean_scores.append(scores.mean())

    return mean_scores

meanL1 = evaluation(1)
meanL2 = evaluation(2)

print(f"{'k':<5}{'L1-Mean':>15}{'L2-Mean':>15}")
for i, k in enumerate(ks):
    print(f"{k:<5}{meanL1[i]:>15.4f}{meanL2[i]:>15.4f}")

plt.figure(figsize=(10, 6))
plt.plot(ks, meanL1, marker='o', label='L1 (Manhattan)', color='blue')
plt.plot(ks, meanL2, marker='s', label='L2 (Euclidean)', color='orange')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Accuracy')
plt.title('KNN Classifier Performance on Digits Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()