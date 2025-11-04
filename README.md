# Taxicab-Geometry
Taxicab Geometry - 2025 창의수학(좌표와 거리의 세계, 10/25) 심화탐구

## 코드 설명

> `코드 설명-코드 제시`의 구조로 서술되어 있습니다.

* 필요한 라이브러리들을 불러옵니다. 그래프 시각화를 위한 matplotlib와 KNN 모델 및 데이터셋 불러오기에 사용할 scikit-learn을 import했습니다.
```py
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
```

* 동점을 만들지 않기 위해 k의 값을 1부터 41까지의 홀수로 설정합니다. 대부분 k의 값의 범위는 1 ≤ k ≤ sqrt(n)으로 정합니다(n은 데이터 개수).
* 분석에 digits(0~9까지의 손글씨 숫자 이미지 데이터셋)를 사용했습니다.
* X는 이미지의 픽셀값으로, 입력 데이터이며 y는 각 이미지의 라벨입니다.
```py
ks = range(1, 42, 2)
digits = load_digits()
X, y = digits.data, digits.target
```

* evaluation 함수를 정의하는 코드입니다. p는 거리 계산 방식을 의미하며, 1일 경우가 맨해튼 거리, 2일 경우는 유클리드 거리입니다. p를 입력받으면 해당하는 거리 계산 방식에서의 각 k값에 대한 평균 정확도 리스트(mean_scores)를 반환합니다.
* 평균 성능 계산을 위해 5-fold Cross-validation 방법을 이용하였습니다. 이에 대해서는 간략하게 후술하겠습니다.
```py
def evaluation(p):
    mean_scores = []

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=p)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        mean_scores.append(scores.mean())

    return mean_scores
```
* 위에서 언급한 evaluation 함수에 p=1을 넣어서 반환된 평균 정확도(mean_scores)를 manhattan이라는 변수에 저장합니다.
* evaluation 함수에 p=2를 넣어서 반환된 평균 정확도를 euclidean이라는 변수에 저장합니다.
```py
manhattan = evaluation(1)
euclidean = evaluation(2)
```
* 결과를 예쁘게(...) 출력합니다. 출력된 결과는 하단 Result 섹션의 표와 같습니다. 높은 정확도는 높은 성능을 의미합니다.
```py
print(f"{'k':<5}{'L1-Mean':>15}{'L2-Mean':>15}")
for i, k in enumerate(ks):
    print(f"{k:<5}{manhattan[i]:>15.4f}{euclidean[i]:>15.4f}")
```
* 위에서 구한 자료를 바탕으로 그래프를 그립니다. 그래프는 하단 Result 섹션에 첨부된 사진과 같이 나옵니다. 파란색 선은 맨해튼 거리의 경우를, 주황색 선은 유클리드 거리의 경우를 나타냅니다.
```py
plt.figure(figsize=(10, 6))
plt.plot(ks, manhattan, marker='o', label='L1 (Manhattan)', color='blue')
plt.plot(ks, euclidean, marker='s', label='L2 (euclidean)', color='orange')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Accuracy')
plt.title('KNN Classifier Performance on Digits Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

## About Cross-validation
![Cross-validation](./img/Cross-validation.png)
1. 데이터를 n등분합니다. 이 프로젝트에서는 사진과 같이 5등분했습니다.
2. 첫 번째 Fold를 테스트 데이터로, 나머지 네 Fold를 학습 데이터로 두고 성능을 평가합니다.
3. 두 번째 Fold를 테스트 데이터로, 나머지 Fold를 학습 데이터로 두고 성능을 평가합니다.
4. 반복.
5. 테스트 데이터에 각 Fold를 한 번씩 넣었으므로 총 다섯 개의 성능 평가 결과를 얻습니다. 이 다섯 값의 평균이 학습 모델의 성능이 됩니다.
<br>
<br>
> 왜 사용하는가?
* `데이터셋이 부족`한 경우.
* 학습 데이터를 과하게 학습시켜 학습 데이터에서는 높은 정확도를 보이나 테스트 데이터에서는 낮은 정확도를 보이는 `과적합` 예방 위함.

## Result

![Result](./img/Performance%20Graph.png)
| k  | L1-Mean | L2-Mean |
|----|---------|---------|
| 1  | 0.9555  | 0.9644  |
| 3  | 0.9566  | 0.9666  |
| 5  | 0.9533  | 0.9627  |
| 7  | 0.9544  | 0.9599  |
| 9  | 0.9561  | 0.9566  |
| 11 | 0.9505  | 0.9555  |
| 13 | 0.9449  | 0.9555  |
| 15 | 0.9455  | 0.9544  |
| 17 | 0.9433  | 0.9533  |
| 19 | 0.9416  | 0.9505  |
| 21 | 0.9410  | 0.9488  |
| 23 | 0.9438  | 0.9483  |
| 25 | 0.9416  | 0.9455  |
| 27 | 0.9394  | 0.9438  |
| 29 | 0.9382  | 0.9410  |
| 31 | 0.9388  | 0.9399  |
| 33 | 0.9366  | 0.9405  |
| 35 | 0.9332  | 0.9366  |
| 37 | 0.9282  | 0.9343  |
| 39 | 0.9277  | 0.9310  |
| 41 | 0.9254  | 0.9299  |