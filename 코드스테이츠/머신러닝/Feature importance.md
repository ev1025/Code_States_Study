# Feature importance(Interpretable ML)
### Mean decrease impurity(MDI)
- 트리기반모델은 각 분기마다 노드의 불순도 감소
- 평균 불순도 감소(MDI) 값을 사용하는 특성중요도
- 빠르고 간편하다는 장점
- high cardinality 특성에 높은 값을 부여함
```python
rf = pipe.named_steps["randomforestclassifier"]
importances = pd.Series(rf.feature_inportances_, x_train.columns)

importances.sort_values().plot.barh() # 특성중요도 플롯
plt.title("")
plt.show();

x_train.unique().sort_values().plot.bar # 카디너리티 중요도
```
### Drop-Columns Importance
- 특정 특성을 drop한 모델과 모든 특성을 사용한 모델을 비교하여 특성의 중요도를 파악
- 평가성능이 하락하면 중요한 특성일 수 있음
- 매번 새로운 학습을 해야하기 때문에 많은 시간이 걸림
```python
score = pipe.score(x_test, y_test)    # 전체 특성의 정확도
print(f" before score : {score:.6f}") 

drop_sample = pd.Series(dtype=float)  # 각 특성drop한 값 넣어줄 곳

for feature in features:              # features 항목 데이터 분리 시에 별도로 만들기
    
    pipe.fit(x_train.drop(columns=[feature], axis = 1), y_train)                 # 특성 드랍한 모델 학습
    score_dropped = pipe.score(x_test.drop(columns=[feature], axis = 1), y_test) # 특성 드랍한 모델의 정확도
    
    print(f"drop {feautre} : {score_dropped:.6f}") # 지운 feature, 평가점수

    drop_sample[feature] = score -  score_dropped  # 전체 - feature를 drop한 모델
```
```python
drop_sample.sort_values().plot.barh() # 특성별 중요도 그래프
plt.title("")
```
### Permutation importances
- 학습된 모델에서 각 feature에 noise를 주어 해당 feature를 사용하지 못하도록(shuffle) 만들고 모델의 성능을 확인
- 모든 모델에 범용적으로 사용가능, MDI보다 high cardinality 특성에 덜 치우침
- 강한 상관관계의 특성이 있으면 잘못된 값을 낼 수 있음
- 비현실적인 결과가 도출될 수 있음 ex) 키2m, 몸무게 30kg
- 특성값의 순서만 변경시키며, 분포는 변화시키지 않음
```python
pi = pd.Series(dtype=float) # permutation 값 넣을 변수    

n_iter = 10                 # 밑에 for문에 넣을 성능평가 횟수

for feature in features:    # 데이터 분할한 features 목록
    x_test_permed = x_test.copy()

    score_permed = []       # pi score값들

    for _ in range(n_iter): 
        x_test_permed[feature] = np.random.permutation(x_test_permed[feature]) # pi 생성
        score_permed.append(pipe.score(x_test_permed, y_test))                 # pi 검증정확도
    
    avg_score = np.mean(score_permed)                       # feature에 noise 준 검증정확도 평균  
    print(f"Permed{feature} : 검증 정확도 {avg_score:.6f}")

    pi = score - avg_score                                   # 특성중요도
```
```python
pi.sort_values().plot.barh() # 특성중요도그래프
plt.title(" ")
```
#### Eli5 라이브러리(permutation 함수)
```python
!pip3 install eli5  # install
import eli5
from eli5.sklearn import Permutationimportance
from sklearn.metrics import roc_auc_score

permuter = PermutationImportance(               # permutation 모델만들기 
    pipe.named_steps['randomforestclassifier'], # 모델
    scoring = 'roc_auc_score',                  # 평가점수
    n_iter = 10,                                # 샘플링 횟수
    random_state = 2
)

x_train_permed = pipe[0].transform(x_train)     # x_train 인코딩
permuter.fit(x_train_permed, y_train)           # permutation 모델 학습
```
```python
pi = pd.Series(permuter.feature_importances_, x_train.columns) # 인덱스 x_train특성인 특성중요도 시리즈
pi.sort_values().plot.barh() # 특성중요도 그래프
plt.title("")
```
```python
eli5.show_weights(
    permuter,
    top = None, # None일경우 전체특성 / top = 사용할 특성 개수
    feature_names = x_train.columns.tolist() # 특성 이름열(꼭 list로 넣어줘야함)
)
```
![캡처](https://user-images.githubusercontent.com/110000734/191256477-47708025-9e0a-4c91-afe3-3ba6e4bbd98b.JPG)
### 특성중요도 선택하는 방법 #1 / #2
```python
#1
feature_names = x_test.columns.tolist() # 학원방식(밑에랑 같은거)
permutation = pd.Series(permuter.feature_importances_, feature_names).sort_values(ascending=False) 
# 인덱스가 feature_names인 permuter모델의 시리즈 생성
features = permutation[(permutation.values > np.mean(permutation.values))].index
# 특성중요도의 평균보다 특성중요도가 높은 feature의 index를 변수 features에 저장


#2
features = pi > pi.mean()         # 내 방식(특성중요도가 평균보다 높은 값)
permed_feats = pi[features].index # 위에서 구한 특성의 인덱스
```
#### 특성중요도 높은 특성으로 roc_auc_score 구하기
```python
from sklearn.metrics import roc_auc_score
rf.fit(x_train[permed_feats], y_train) # 특성중요도 높은 특성으로 모델 재학습

x_train_pred_proba = rf.predict_proba(y_train, x_train[permed_feats])[:,0]
# predict_proba는 0, 1일 확률을 각각 구함([:,0] 0일 확률 배열 / [:, 1] 1일 확률 배열
roc_auc_score(y_train, x_train_pred_proba)
```
