# Boosting

### Bagging vs Boosting
- 앙상블 : 여러 기본 모델을 학습하고 모델들의 예측값을 합하여 최종 예측을 하는 방법
- 단일 모델 사용 기법의 과적합, 과소적합을 해소하기 위해 사용됨
- 대표적인 앙상블 기법이 배깅과 부스팅
#### Baging
- 좁은 의미 : 복원추출 - 기본모델학습(weak learner) - weak learner 합치기
- 넓은 의미 : 기본모델(weak learner)을 병렬(parallel)로 학습하고 평등하게 예측값을 합치는 과정
- 기본모델들이 상호영향 없이 독립적, 병렬적으로 학습되며 RandomForest가 주요 모델
- 여러 기본모델들이 서로 오차를 상쇄시켜 분산을 줄이고 과적합 해소

#### Boosting
- 기본 모델들이 순차적으로 학습하는 방법(sequential)
- 앞의 모델에서 잘 예측하지 못하는 부분을 집중해서 학습
- AdaBoost, Gradient Boosting 모델
- 반복할수록 최종모델의 복잡도가 상승하며 편향을 줄여 과소적합을 해소
- 앞모델에서의 오류값*learning_rate를 더하여 누적학습 진행 -> learnig_rate가  높으면 과적합이 되기 쉽다.   
- 러닝레이트 시각자료   
<http://arogozhnikov.github.io/2016/07/05/gradient_boosting_playground.html>
- 0값에 가까울수록 오차가 적다(=과적합되었다)   
![Untitled](https://user-images.githubusercontent.com/110000734/190153132-c1d80297-935b-403a-81e8-671da38a48a2.png)



---
### AdaBoost
- 분류문제에 사용되며, 이상치에 민감하고 성능이 떨어짐
- 첫 모델이 잘못 분류한 관측치를 다음 모델이 샘플링 할 확률을 높임(가중치를  줌)
- 두번째 모델이 잘못 분류한 관측치를 다음 모델이 샘플링 할 확률을 높힘
- 최종예측 시 각 기본모델의 가중치르 다르게 주어 예측   
![AdaBoost](https://user-images.githubusercontent.com/110000734/190152995-63db8b93-b808-4c5d-804c-c0c5a4a1ba53.png)




### Gradient Boosting
- 회귀, 분류 문제에 모두 사용
- 틀린데이터에 집중하기위해 잔차를 학습(다음 모델이 이전 모델의 잔차 학습)
- 잔차학습은 잔차가 큰 관측치를 더 학습하도록하여 이전 모델을 보완
- scikit-learn Gradient Tree Boosting — 상대적으로 속도가 느림
  - Anaconda: already installed
  - Google Colab: already installed
- LightGBM — 결측값을 수용하며, monotonic constraints를 강제할 수 있습니다.
  - Anaconda: conda install -c conda-forge lightgbm
  - Google Colab: already installed
- CatBoost — 결측값을 수용하며, categorical features를 전처리 없이 사용할 수 있습니다.
  - Anaconda: conda install -c conda-forge catboost
  - Google Colab: pip install catboost

### XG Boost
- gradient boosting 모델 중 하나
— 결측값을 수용하며, monotonic constraints를 강제할 수 있다.(비단조 모델을 단조모델로 바꿔줄 수 있음)
- 민감하고 노이즈가 많으면 작업하기 어려움 -> 하이퍼파라미터가 많다.
- 범주형 특성을 수치화한다.(더미변수, 인코딩)
- 특성가중치가 모두 같기 때문에 normalization, scalering을 하여도 값의 변화가 없다.
- ordinal encoding을 사용함
  - Anaconda, Mac/Linux : conda install -c conda-forge xgboost
  - Windows : conda install -c anaconda py-xgboost
1. 데이터셋을 나눔 train val test
```python
!pip install category_encoders    # 카테고리 인코더 설치
from sklearn.model_selection import train_test_split
train, val = train_test_split(train, test_size=0.2, stratify = train[target], random_state=2)
```
2. EDA, Feature engineering
```python
# Feature Engineering을 수행합니다.
def engineer(df):
    # 새로운 특성을 생성합니다.
    behaviorals = [col for col in df.columns if "behavioral" in col]
    df["behaviorals"] = df[behaviorals].sum(axis=1)

    # 사용하지 않는 특성을 drop합니다.
    dels = [col for col in df.columns if ("employment" in col or "sess" in col)]
    df.drop(columns=dels, inplace=True)

    return df


train = engineer(train.copy())
val = engineer(val.copy())
test = engineer(test.copy())
```
3. feature와 target 분리
```python
def xy_split(df):
    x = df.drop(target, axis=1)
    y = df[target]
    return x, y

x_train, y_train = xy_split(train)
x_val, y_val = xy_split(val)
x_test = test
```
4-1. 모델 학습(make_pipeline)
```python
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(                 # 파이프라인 만들기(ealry_stopping 불가)
    OrdinalEncoder(),
    SimpleImputer(strategy="mean"),   # 결측치를 평균값으로 채움
    XGBClassifier(        
        booster = 'dart',             #'gbtree' : decision tree 사용 / 'dart' : 결정트리를 사용하되, 과적합 방지를 위해 일부 트리를 drop / 'gblinear' : 선형모델(지양)
        objective = 'binary:logistic',# 비용함수(목적함수)/ 이진분류  / reg:squarederror = MSE 최소화 회귀 / reg:logistic = Logistic 회귀
        eval_metric='error',          # error = 1 - accuracy 지표를 사용해 평가 / regression: rmse, classification: logloss
        n_estimator = 1000,           # 기본모델(weak learner)의 개수
        random_state = 2,
        n_jobs=-1,
        max_depth = 6,                # 분기의 개수 / 너무 크면 과적합(일반적으로 5~12)
        learning_rate = 0.2           # 기본모델의 반영정도(0~1) / 보통 0.05~0.3 / 너무크면 과적합, 너무 작으면 학습이 느려짐
    )
)
print(pipe) # pipe 항목들 출력
```

4-2. 모델검증
```python
from sklearn.metrics import accuracy_score, f1_score
pipe.fit(x_train, y_train)            # 모델 훈련

x_train_pred = pipe.predict(x_train)  # 훈련 데이터 예측값
x_val_pred = pipe.predict(x_val)      # 검증 데이터 예측값

print(f"훈련 : {f1_score(y_train, x_train_pred)}")
print(f"검증 : {f1_score(y_val, x_val_pred)}")
```
5-1. pipeline모델
```python
from sklearn.pipeline import Pipeline              # 파이프라인 만들기(ealry_stopping 가능)

model = Pipeline(steps=[('ordinaryencoder', OrdinalEncoder()),              # 인코딩
                        ('simpleimputer', SimpleImputer(strategy='mean')),  # 평균으로 채우기
                        ('xgbclassifier', XGBClassifier(                    # XGB 분류기
                            random_state = 2,              
                            n_estimators = 1000,                            # 트리 개수
                            max_depth = 6,                                  # 분기 개수(나눠지는것)
                            objective = "binary:logistic",                  # 비용함수
                            n_jobs = -1,                                    # 자유도
                            learning_rate = 0.2,                            # 0.1~0.3 (누적으로 진행됨)
                            eval_metric = "error"                           # error = 1 - accuracy 지표를 사용해 평가 / regression: rmse, classification: logloss
                        ))])

enc = model[0]                            # ealry_stopping 할  데이터 인코딩
x_train_enc = enc.fit_transform(x_train)
x_val_enc = enc.transform(x_val)

imp = model[1]                            # ealry_stopping 할  데이터 결측치 제거
x_train_enc = imp.fit_transform(x_train_enc)
x_val_enc = imp.transform(x_val_enc)

early_list = [(x_train_enc, y_train), (x_val_enc, y_val)] # ealry_stopping 할 데이터(eval_set)

model.fit(        # pipeline 모델
    x_train,
    y_train,
    xgbclassifier__eval_set = early_list,                 # (훈련데이터, 검증데이터)리스트의 오류값
    xgbclassifier__early_stopping_rounds = 200            # 최적의 트리개수 찾아줌(200회까지 반복해서 성능 향상 안되면 n_estimator 중단)
)
```
5-2. pipeline 검증
```python
x_train_pred_e = model.predict(x_train)
x_val_pred_e = model.predict(x_val)

print(f"훈련 : {f1_score(y_train, x_train_pred_e)}")
print(f"검증 : {f1_score(y_val, x_val_pred_e)}")
```
### 구글드라이브 마운트, 파일 저장
마운트
```python
# csv 파일 저장을 위한 구글 드라이브 마운트
import sys

if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
```
파일 저장
```python
y_pred_test = model.predict(x_test)
sample_submission['vacc_h1n1_f'] = y_pred_test # 열추가
sample_submission.to_csv("/content/drive/MyDrive/sample_submission.csv", index = False) # sample_submission.csv 파일 마이드라이브에 저장
```
### XGBoost파라미터

#### `booster`
- weak learner 모델을 설정할 수 있는 파라미터입니다.
- `gbtree`: Decision Tree 모델을 사용합니다.
- `dart`: Decision Tree 모델을 사용하되, DART 알고리즘을 사용하여 모델을 정규화합니다. 
  - 과적합을 방지하기 위해 이전에 학습된 트리 중 몇 가지를 drop시키는 기법입니다.
  - 관련 정보는 [여기](https://xgboost.readthedocs.io/en/latest/tutorials/dart.html)를 참조하세요.
-  `gblinear`: 선형 모델을 사용합니다. 표현력이 제한적이어서 잘 사용되지 않습니다.

#### `objective`
- 최소화하고자 하는 목적함수를 설정할 수 있습니다.
- 설정하지 않을 경우, 분류 / 회귀 Task에 따라 설정된 기본값으로 설정됩니다(`XGBClassifier`:  `binary:logistic`, `XGBRegressor`:  `reg:squarederror`,).
- `reg:squarederror`: MSE를 최소화하여 회귀 문제를 해결합니다.
- `reg:logistic`: Logistic 회귀 문제를 해결합니다.
- `binary:logistic`: Logistic 이진분류 문제를 해결합니다.
- 이 외에도 다양한 objective를 지원합니다. [여기](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)를 참조하세요.


#### `eval_metric`
- 검증 데이터를 같이 넣어줄 경우, 검증 방법을 설정할 수 있습니다.
- 설정하지 않을 경우 `objective`에 따라 설정된 기본 `eval_metric`으로 설정됩니다(`regression`: `rmse`, `classification`: `logloss`).


### GBDT(XGBoost)의 주요 하이퍼파라미터

- `n_estimators`
  - weak learner들의 수를 결정합니다.
- `learning_rate`
  - 단계별로 weak learner들을 얼마나 반영할지 결정합니다.
  - 0~1의 범위를 갖습니다
    - 값이 너무 크면 과적합이 발생하기 쉽습니다.
    - 값이 너무 작으면 학습이 느려집니다.
  - 일반적으로 0.05 ~ 0.3 정도의 범위에서 탐색을 진행합니다.
- `max_depth`
  - 각 weak learner 트리들의 최대 깊이를 결정합니다.
  - 모델의 성능에 가장 큰 영향을 주는 변수입니다.
  - 0 ~ ∞의 범위를 갖습니다.
    - -1으로 설정 시 깊이의 제한이 없습니다.
    - 값이 너무 크면 과적합이 발생하기 쉬우며 메모리 사용량이 늘어납니다.
  - 일반적으로 5 ~ 12 정도의 범위에서 탐색을 진행합니다.
- `min_child_weight` 
  - leaf 노드에 포함되는 관측치의 수를 결정합니다.
  - 0 ~ ∞의 범위를 갖습니다.
    - 값이 커질수록 weak learner들의 복잡도가 감소합니다.
  - 일반적으로 과적합 발생 시 1, 2, 4, 8...와 같이 값을 2배씩 늘려 성능을 확인합니다.
- `subsample`
  - 각 weak learner들을 학습할 때 과적합을 막고 일반화 성능을 올리기 위해 전체 데이터 중 일부를 샘플링하여 학습합니다.
  - subsample 파라미터가 데이터(row)를 샘플링할 비율을 결정합니다.
  - 0 ~ 1의 범위를 갖습니다.
  - 일반적으로 0.8 정도로 설정하며, 데이터의 크기에 따라 달라질 수 있습니다.
- `colsample_bytree`
  - 각 weak learner들을 학습할 때 과적합을 막고 일반화 성능을 올리기 위해 전체 column 중 일부를 샘플링하여 학습합니다.
  - `colsample_bytree` 파라미터가 column을 샘플링할 비율을 결정합니다.
  - 0 ~ 1의 범위를 갖습니다.
  - 일반적으로 0.8 정도로 설정하며, 특성의 갯수에 따라 달라질 수 있습니다. 특성이 천 개 이상으로 매우 많을 경우 0.1 등의 매우 작은 값을 설정하기도 합니다.
- `scale_pos_weight`
  - scikit-learn의 class_weight와 동일한 기능입니다. 
  - `sum(negative cases)` / `sum(positive cases)` 값을 넣어 주면 scikit-learn의 `balanced` 옵션과 동일하게 됩니다.
  - imbalanced target일 경우 적용을 고려합니다.
  


  
- 일반적으로 `max_depth`와 `learning_rate`가 가장 중요한 하이퍼파라미터로 다뤄지며, 과적합을 방지하기 위해 `subsample`, `colsample_bytree` 등의 값을 추가로 조정해 줍니다.
- 이 외에도 XGBoost는 굉장히 많은 하이퍼파라미터를 제공합니다. [여기](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster)를 참조하세요.

#### Early Stopping
- n_estimators를 최적화 해주는 파라미터(다른 값을 변경할 때 따로 바꿔주지 않아도 됨)
- XGBoost라이브러리에서 제공
```python
watchlist = [(X_train_encoded, y_train), (X_val_encoded, y_val)] # 기준데이터셋

model.fit(
    X_train_encoded,
    y_train,
    eval_set=watchlist,          # 기준데이터셋 제공(여러개면 마지막 데이터)
    early_stopping_rounds=50,  # 50 rounds 동안 성능 개선이 없으면 학습을 중지합니다.
)
```
