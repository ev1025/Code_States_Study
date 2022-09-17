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

## Bayesian Search
- 이전에 탐색한 조합들의 성능을 기반으로 성능이 좋은 조합을 선택
- 효율적으로 탐색하여 한정된 자원 내의 좋은 조합 발견가능
```python
hp.choice(label, option)         # 리스트나 튜플형태 하나를 선택("strategy", ["mean","median"])
hp.randint(label, upper)         # 범위 내의 정수값을 랜덤으로 선택 0~upper값 랜덤정수0 ("randint", upper)
hp.uniform(label, low, high)     # 범위 내의 실수 값을 랜덤으로 선택
hp.quniform(label, low, high, q) # 범위 내에 q간격에서 랜덤수 선택
hp.normal(label, mu, sigma)      # 평균, 표준편차를 갖는 정규분포에서 실수값을 랜덤으로 선택
```
- 파라미터 설정
```python
!pip3 install hyperopt
from hyperopt import hp

params = {
    "xgbclassifier__strategy" : hp.choice('strategy', ['mean', 'median']) # 평균이나 중간값 랜덤으로 사용
    "xgbclassifier__maxdepth" : hp.quniform('max_depth', 2 10, 2)         # 2~10 사이의 값을 2step으로 균등분포로  선택
    "xgbclassifier__min_child_weight" : hp.quniform('min_child_weight', 2, 10 ,2) 
    "xgbclassifier__colsample_bytree" : hp.uniform('colsample_bytree', 0.5, 1.0)
}
```
#### 사용방법
- hyperopt.fmin : 주어진 함수의 loss정보를 이용하여 다음 하이퍼파라미터 조합을 선택
- loss를 가장 작게 만드는 파라미터 조합을 선택해서 클수록 좋은 metrics는 - 부호 붙여주기
```python
from hyperopt import fmin, tpe, Trials, STATUS_OK  
from sklearn.model_selection import cross_val_score
from category_encoders import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def get_pipe(params):    
    params['xgbclassifier_max__depth'] = int(params['xgbclassifier__max_depth'])  # 정수형으로 변경
    model = make_pipeline(
        OrdinalEncoder(),
        SimpleImputer(strategy ='mean'),
        XGBClassifier(
            objective ="binary:logistic",
            eval_metric = "error",
            n_estimator = 200,
            random_state = 42,
            n_jobs = -1,
            learning_rate = 0.1,
            use_label_encoder = False
        )
    )
    model = model.set_params(**params) # model에 위의 파라미터를 적용시킴
    return model

def fit_and_eval(params):              # cv검정 함수
    model = get_pipe(params)            
    score = cross_val_score(model, x_train, y_train, cv=3, scoring='f1_score')
    avg_cv_score = np.mean(score)
    return {"loss" : -avg_cv_score, "status" : STATUS_OK} # f1_score는 높을수록 좋아서 - 붙임, status는 상태확인


trials =(                  # fmin의 기록 저장
    Trials()
)

best_params = fmin(        # 위에서 만든 함수로 최적의 하이퍼파라미터를 찾아주기
    fn = fit_and_eavl      # 위에 만든 함수 
    trials = trials,       # 탐색 기록 trials에 저장
    space = params,        # 사용할 하이퍼파라미터 목록
    algo = tpe.sugegest,   # 탐색할 알고리즘 방식
    max_evals = 10         # 10회 탐색(= n_estimator)
)
```
```python
trials.trials              # 모든 trial 정보에 접근
```
```python
trials.best_trials['misc'],['vals']   # .best_trials만 가져오기(misc,vals는 위 trials의 인덱스)
-trials.best_trials['result']['loss'] # result인덱스의 loss를 가져옴(f1_score는 높을수록 점수가 좋아서 음수표시)
```

# 특성 중요도
- 선형회귀모델 : coefficient
- 트리기반모델 : feature_importance`
```python
from sklearn.model_selection import train_test_split

train, val = train_test_split(train, test_size = 0.2, stratify=train[target], random_state=2)
x_train, y_train = train.drop(columns = target), train[target]
x_val, y_val = val.drop(columns=target), val[target]

pipe = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(strategy="mean"),
    XGBClassifier(
        object = "binary:logistc",
        eval_metric="f1_score",
        n_estimators = 200,
        random_state = 42,
        n_jobs = -1,
        max_depth = 6,
        min_child_weight = 6,
        colsample_bytree = 0.7,
        learning_rate = 0.1,
        use_label_encoder = False,
    ),
)

pipe.fit(x_train, y_train)
pipe.score(x_val, y_val)  # fi_score 확인
```
```python
mport matplotlib.pyplot as plt  # 특성중요도 표

feature_importances = list(
    zip(X_train.columns, pipe.named_steps["xgbclassifier"].feature_importances_)
)
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 10), dpi=120)
plt.barh(*list(zip(*feature_importances[::-1])))
plt.axvline(0.015, color="red", linewidth=0.5)
```
## 통계량기반 특성
- 모델학습 전 특성과 타겟 간 통계량을 확인하여 특성을 제거할 수 있음
### 피어슨 상관계수 
- 연속성을 지닌 특성과 타겟의 상관계수 추측가능
- np.corrcoef
### 스피어만 상관계수
- 단조성을 지닌 특성과 타겟의 상관계수 추측(한 특성이 커질 때 다른특성이 커지는지)
- 특성의 값들을 전체에서의 순위로 대치한 후 피어슨 상관계수를 구함
- scipy.stats.speamanr
```python
from scipy.stats import spearmanr

np.corrcoef(x, y)[0, 1] # 두 특성  x, y의 피어슨 상관계수
spearmanr(x, y)         # 두 특 성 x, y의 스피어만 상관계수
```
### SelectKBest
- 문제의 유형(분류, 회귀)에 따라 다양한 통계량을 기반으로 특성을 선택
- sklearn.feature_selection.SelectKBest (score_func인자를 변경)
- 분류문제 : f_classif / mutual_info_classif / chi
- 회귀문제 : f_regression / mutual_info_regression
```python
from category_encoders import OneHotEncoder # 전처리

enc = OrdinalEncoder()
imp = SimpleImputer()

X_train_encoded = enc.fit_transform(X_train)
X_train_imputed = imp.fit_transform(X_train_encoded)

X_val_encoded = enc.transform(X_val)
X_val_imputed = imp.transform(X_val_encoded)
```
```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(score_func = mutual_info_classif, k=6)      # 분류인지,회계인지 / 특성의 개수
x_train_selected = selector.fit_transform(x_train_imputed, y_train)
x_val_selected = selector.transform(x_val_imputed)
```
```python
print("선택된 특성: ", X_train_encoded.columns[selector.get_support()].tolist()) # 사용된 특성 출력
.selector.get_support # 모델이 사용한 컬럼 Bool값으로 표현
.tolist()             # 값을 일자로 나열함(데이터 보기 편하게)
# 후에 X_train_selected 로 pipe모델 학습하여 model.score(X_val_selected, y_val) 확인하고 비교하여 적절한 k값 찾기
```


### 함수
```python
enc = pipe.named_steps['ordinalencoder'] # 파이프라인의 ordinalencoder 사용
enc.transform(x_val)                      # 요런식으로 모델로 사용

pipe.named_steps['xgbclassfier'].feature_importances_ # 특성중요도

.tolist()  # 값들을 하나하나 볼 수 있도록 1줄로 표시해줌
```
