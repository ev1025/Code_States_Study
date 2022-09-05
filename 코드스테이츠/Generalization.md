# Generalization(일반화)
- 훈련된 모델이 새로운 데이터에 대해서도 적절하게 작동하는 것

### Hold-Out
- 데이터셋을 학습 데이터와 평가데이터로 나누는 것
- 훈련(Traning), 검증(Validation), 평가(Test) 데이터로 나누는 방법(3 way hold- out)
```python
from sklean.model_selection import train_test_split                    # 데이터 나누기(hold-out)

train, test = train_test_split(df, test_size =0.25, random_statae=42)  # df에서 train과 test 데이터로( 75% : 25%) 나누기
train, val = train_test_split(train, test_size=0.25, random_state=42)  # train에서 train과 val 데이터로 (75% : 25%) 나누기


num_features = df.dtypes[df.dtypes!='object'].index                        # 수치데이터 변수저장
df[num_features].corr()['SalePrice'].sort_values(ascending=False).head(10) # 상관계수 구하기


target = 'SalePrice'  # 타겟 추출
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'FullBath'] # 특성 추출


def x_y_split(df) :          # 특성(X), 타겟(y) 함수
    X = df[features]
    y = df[target]
    return X, y

X_train, y_train = x_y_split(train)    # 훈련데이터 특성, 타겟 
X_val,   y_val   = x_y_split(val)      # 검증데이터 특성, 타겟 
X_test,  y_test  = x_y_split(test)     # 테스트데이터 특성, 타겟
 
 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score   # 회귀평가지표

def eval_models(y_true, y_pred) :
    mse = mean_squared_error(y_true, y_pred).round(3)
    rmse = np.sqrt(mse).round(3)
    mae = mean_absolute_error(y_true, y_pred).round(3)
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mae, r2

baseline = [y_train.mean()] * len(y_train)  # 기준모델의 예측값

from sklearn.linear_model import LinearRegression

model = LinearRegression()                # 선형회귀 변수 생성
model.fit(X_train, y_train)               # 훈련 모델 생성

y_train_pred = model.predict(X_train)     # 훈련데이터 예측값
y_val_pred = model.predict(X_val)         # 검증데이터 예측값
y_test_pred = model.predict(X_test)       # 테스트데이터 예측값


comparison = pd.DataFrame(index=['mse', 'rmse', 'mae', 'r2'], columns=['Base', 'Train', 'Validation', 'Test'])  # 회귀평가지표 데이터프레임 생성
comparison['Base'] = eval_models(y_train, baseline)       # 기준모델 (관측값, 예측값)
comparison['Train'] = eval_models(y_train, y_train_pred)  # 훈련모델 (관측값, 예측값)
comparison['Validation'] = eval_models(y_val, y_val_pred) # 검증모델 (관측값, 예측값)
comparison['Test'] = eval_models(y_test, y_test_pred)     # 테스트모델(관측값, 예측값)
comparison
```

### Cross-Validation
- cv, 교차검증이라고 하며 모든 샘플을 검증에 사용
- 3way hold-out은 데이터가 적을 경우 적절하지 않음 (랜덤값에 따라 편차가 심할 수 있음) -> cv 사용
- 일반적으로 많이 사용되는 방법은 k-fold cross-validation (데이터를 모두 사용해 k번 학습과 검증을 반복)
 1. Train 데이터와 Test 데이터 분리
 2. 데이터를 k개로 분할합니다.
 3. 1개의 파트는 검증세트로 이용되고 나머지 k-1개의 파트는 모두 훈련에 사용  4. k번의 검증 결과를 종합하여 전체 모델의 성능을 평가

#### k-fold
```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size =0.25, random_statae=42)  # df에서 train과 test 데이터로( 75% : 25%) 나누기

target = 'SalePrice'
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'FullBath']

def x_y_split(df) :         # 타겟, 특성함수
    X = df[features]
    y = df[target]
    return X, y

X_train, y_train = x_y_split(train)  # 특성, 함수 분리
X_test, y_test = x_y_split(test)

from sklearn.model_selection import KFold     # k-fold 불러오기
kf = KFold(n_splits=5)                        # 5집단으로 분리
kf.get_n_splits()                             # kf의 개수 확인가능(선택사항)

cv_result = []         

for train_idx, test_idx in kf.split(X_train) :   # X_train데이터를 4:1로 나누어 인덱스에 저장
    X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]

    model.fit(X_train_cv, y_train_cv)    # 위에 Hold-out 모델 훈련
    y_pred_cv = model.predict(X_val_cv)  # 검증데이터 예측

    mae_cv = mean_absolute_error(y_val_cv, y_pred_cv).round(2) # 검증데이터의 mae
    cv_result.append(mae_cv)
```
#### cross validation
- 쉽게 k-fold 하는 방법
```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, random_state=2 )  # 훈련데이터 80 : 시험데이터 20

feature = ['bedrooms', 'bathrooms', 'sqft_living']     # 특성 추출
target= 'price'                                        # 타겟 추출

def x_y_split(df):                                     # 특성, 데이터 생성 함수

    X = df[feature]                                  
    y = df[target]
    return X, y

train_x, train_y = x_y_split(train)                    # 훈련 데이터 X,y
val_x,    val_y  = x_y_split(test)                     # 시험 데이터 X,y


model = LinearRegression()                             # 훈련 데이터 선형회귀
model.fit(train_x, train_y)                            # 모델 훈련

train = model.predict(train_x)                         # 모델로 훈련데이터 예측

from sklearn.model_selection import cross_val_score    # Cross validation 
cv_result = cross_val_score(model, train_x, train_y, cv=10, scoring="neg_root_mean_squared_error")  # k-fold 10개 rmse 값
                             모델   특성     타겟     k 수        사용할 회귀지표(neg = 마이너스(낮아야 좋기때문에))
```
- scoring 인자값 https://scikit-learn.org/stable/modules/model_evaluation.html   


### 과적합(overfitting), 과소적합(underfitting)
- 일반화 오차 : 테스트데이터의 오차
- 테스트데이터에서도 좋은 성능을 내면 일반화 잘 된 모델
> 과적합
  - 훈련데이터의 디테일과 노이즈까지 모두 학습 -> 새로운 데이터 예측x
  - 유연성이 높은 non-parametric모델이나 비선형 모델에서 나타날 가능성이 높다.
> 과소적합
  - 훈련 데이터셋도 제대로 학습하지 못해서 새로운 데이터를 잘 예측하는 못하는 현상
![분산](https://user-images.githubusercontent.com/110000734/188460420-8591d2c3-890e-4505-89bb-965a7b92dd68.png)


#### 편향 
- 모델의 예측값과 실제값의 차이
- 잘못된 알고리즘, 모델의 가정에서 오는 에러
- 단순선형회귀모델 > 다항 선형회귀모델의 차이 (편향이 더 큼)
- 편향이 큰 모델은 과소적합이 일어날 수 있음
#### 분산
- 새로운 데이터에 대한 모델의 예측값 변동양
- 분산이 큰 모델은 훈련데이터의 노이즈까지 학습하여 과적합이 일어날 수 있음

### 분산과 편향의 Trade-off
- MSE 식을 reducible, irreducible 에러로 나누어 표현하면 Bias 에러 + Variance 에러 + irreducible 에러로 나뉘게 됩니다.
$${\displaystyle \operatorname {E} _{D}{\Big [}{\big (}y-{\hat {f}}(x;D){\big )}^{2}{\Big ]}={\Big (}\operatorname {Bias} _{D}{\big [}{\hat {f}}(x;D){\big ]}{\Big )}^{2}+\operatorname {Var} _{D}{\big [}{\hat {f}}(x;D){\big ]}+\sigma ^{2}}$$



### 오늘의 함수
```python
df.sample(frac= 0~1) # 0~100% 만큼의 데이터 샘플로 추출

features = df.dtypes[df.dtypes!='object'].index   # object가 아닌 데이터 인덱스 저장
df[features]                                      # df에features 인덱스를 가진 데이터)
```

