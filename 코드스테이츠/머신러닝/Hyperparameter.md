# Hyperparameter tunig
### 주요 하이퍼 파라미터를 3개 이내로 먼저 조정
- 회귀모델 : alpha, C
- 트리기반모델 : max_depth, min_samples_split
- 부스팅모델(GBDT) : max_depth, learning_rate
  - max_depth : 복잡도를 증가시킴
  - min_samples_split / min_sample_leaf : 복잡도를 낮춰줌   

## Exhaustive Grid Search
- 지정한 범위 내의 모든 조합을 학습하여 조합 선택
- 검증결과를 바탕으로 하이퍼파라미터의 영향을 직관적으로 이해할 수 있음
- 하이퍼파라미터의 수가 많을 때는 효율적이지 못함

```python
params = {
    "xgbclassifier__max_deps" : [2, 4, 6],                # 각 하이퍼파라미터 값 지정
    "xgbclassifier__min_child_weight" : [2, 4, 8],        # 각 3 개씩 3x3x3
    "xgbclassifier__colsample_bytree" : [0.6, 0.8, 1.0]
}
```
```python
from sklearn.model_selction import GirdSearchCV

grid_serach = GridSearchCV(model,                # 훈련 모델
                            param_grid = params, # 지정한 파라미터
                            scoring="metrics",   # 사용할 평가지표
                            cv = n,              # 교차검증 횟수
                            verbose=1~3)         # 정보표시(1 : 계산시간 / 2 : 점수까지 / 3 : 매개변수 인덱스까지 )
grid_search.fit(x_train, y_train)
```
```python
print("최적 하이퍼파라미터: ", grid_search.best_params_) # best_params_
print("최적 metrics_score: ", grid_search.best_score_)   # best_score_
```
```python
# rank_test_score : 테스트 순위
# mean_score_time : 예측에 걸리는 시간
# model.cv_results_ : cv결과데이터
pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score").T
```
## Randomized Search
- 지정한 범위 내의 랜덤 조합을 학습하여 조합 선택
- 탐색 횟수를 지정해주어 빠르게 탐색가능
- 랜덤이라서 이상적인 결과 안나올 수 있음
- 범위로 값을 지정하거나, scipy.stats의 분포로 지정 가능
- 하이퍼파라미터의 범위가 넓어도 fitting해준 만큼만 진행됨(n_iter)
```python
from scipy.stats.distributions import uniform

params = {
    "simpleimputer__strategy" : ["median", "mean"],
    "xgbclassifier__max_depth" : [2, 4, 6],
    "xgbclassifier__min_child_weight" : [2, 4, 8],
    "xgbclassifier__colsample_bytree" : uniform(  
        loc = 0.5, scale = 0.5                      # 0.5 ~ 1 값을 균등분포로 범위지정
    )
}
```
```python
from sklearn.mdoel_selection import RandomizedSearchCV

randomized_search =RandomizedSearchCV(
    model,
    param_distributions = params,  # 파라미터 목록
    scoring = "metrics",
    n_iter = 10,                   # 반복횟수(default = 10) / 런타임과 솔루션의 품질을 절충
    cv = 3,                        # 교차검증횟수
    verbose = 3,                   # 정보표시(1 : 계산시간 / 2 : 점수까지 / 3 : 매개변수 인덱스까지 )
    random_state = 42
)
randomized_search.fit(x_train, y_train)
```
```python
print("최적 하이퍼파라미터: ", randomized_search.best_params_) # .best_params_
print("최적 metrics: ", randomized_search.best_score_)             # .best_score_
```
```python
# rank_test_score : 테스트 순위
# mean_score_time : 예측에 걸리는 시간
# model.cv_results_ : cv결과데이터
pd.DataFrame(randomized_search.cv_results_).sort_values(by="rank_test_score").T
```
## Bayesian Search
- 이전에 탐색한 조합들의 성능을 기반으로 성능이 좋은 조합을 선택
- 효율적으로 탐색하여 한정된 자원 내의 좋은 조합 발견가능
```python
!pip3 install hyperopt

```
