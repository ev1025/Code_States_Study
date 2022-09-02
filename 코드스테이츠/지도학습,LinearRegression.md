# 지도학습
### 1. 데이터
- 특성(feature) : 독립변수(x), 라벨을 예측하기 위해 사용하는 데이터 (털 길이, 색상, 다리 길이..)
- 라벨(target) : 종속변수(y), 예측모델의 정답 (스핑크스, 러시안블루, 코숏..),             
- 데이터 사이즈 : 데이터의 양(1000마리의 데이터)
- 데이터 다양성 (Diversity) : 데이터의 종류( 스핑크스, 러시안블루, 코숏)

### 2. 모델
- 특성과 타겟의 수학적 관계를 정의한 것

### 3. 학습(Trainig)
- 라벨이 있는 데이터를 학습한다.
- 특성과 타겟의 수학적 관계를 찾아내어 새로운 타겟을 잘 예측하는 것

### 4. 평가(Evaluation)
- 모델의 학습을 확인하고 평가
- 학습과 검증의 과정이 더 필요한지 모델을 배포할지 결정

### 5. 추론(inference)
- 새로운 데이터의 예측을 수행

### 회귀 vs 분류
- 회귀 : 연속적 인 값을 예측하는 문제 (주택가격, 강수량)
- 분류 : 데이터의 특정 범주에 속할 확률 (이진분류, 3개 이상의 분류)
- 내가 어떤 분류를 하고 있는지 정의하는 것이 중요하다. (두 가지의 과정이 다름)

## Linear Regression(선형회귀모델)
### 데이터 분석
1. 훈련데이터와 평가데이터를 나눈다. 평가데이터의 타겟값은 제거
2. 훈련데이터의 x축,y축,타겟 컬럼, 평가데이터의 x축,y축 컬럼 생성
3. 타겟의 기준점을 정해서 기준점, 전체 describe 진행
4. displot으로 시각화해서 살펴보기(mean, median)
5. scatterplot으로 훈련데이터 시각화 후 
6. 기준모델 설정(보통 평균이나 중간값)
sns.lineplot(x=df['GrLivArea'], y=df['SalePrice'].mean(), color='red')

### Linear Regression(선형회귀모델)
- 잔차(residual) : 실제값(y) – 예측값(y^)  => y – ax+b # 모집단에서는 오차(Error)라 함
- 예측값 : y^ = ax+b (알파,베타)
- RSS(residual sum of squares) : 잔차 제곱의 합(y-y^)**2, 선형회귀모델의 비용함수, 비용함수를 최소화 하는 알파와 베타를 찾는 것
- OLS(Ordinary Least Square) : 최소제곱법 (잔차의 제곱을 최대한 작게 만들어줌)
- 선형회귀그래프
  - x = 독립변수 : 예측변수(Predictor), 설명변수(Explanatory), 특성(feature) 
  - y = 종속변수 : 반응변수(Response), 레이블(Label), 타겟(Target)

### Scikit-Lean을 이용한 선형회귀모델
- 특성데이터와 타겟데이터로 나누어줌
  - 특성행렬(X) = [‘n_samples’, ‘n_features’] # 보통 2차원 행렬 np or dataframe
  - 타겟배열(y) = ‘n_samples’ # 보통 1차원 배열 np or series
  - 보간(interpolate) : 모델이 학습한 데이터 구간 안에 있는 값을 예측
  - 외삽(extrapolate) : 모델이 학습한 데이터 구간 바깥에 있는 값을 예측

### 단순선형회귀 (Simple Linear Regression)
```python
from sklearn.linear_model import LinearRegression
feature_s = ['sqft_living']   # 특성(feature) 변수지정
target_s = 'price'            # 타겟(label) 변수지정

X = df[feature_s]
y = df[target_s]

simple_ols = LinearRegression() # 선형회귀 변수지정
simple_ols.fit(X,y)             # 선형회귀 훈련
simple = simple_ols.predict(X)  # 훈련된 모델로 특성 지정하여 타겟 예측(타겟 목록)

sns.scatterplot(data = df, x = 특성, y = 타겟)  # 데이터 분포 확인 (y에 예측값 넣으면 선형회귀선처럼 나옴)

sns.regplot(x = 특성, y = 타겟)                 # 스캐터 + 선형회귀 그래프
```
### 다중선형회귀
- 두 개 이상의 독립변수로 종속변수를 설명하는 선형회귀
- y = a1x1+a2x2+...anxn +b(알파,베타)
- scatterplot(x,y,hue,size=df[]) 로 2차원에 3차원표시가능
```python
from sklearn.linear_model import LinearRegression 

feature_m = ['sqft_living','bathrooms']   # 다중 특성값 추출
target_m = 'price'                        # 타겟값 추출

X_m = df[feature_m]
y_m = df[target_m]

multi_ols = LinearRegression()             
multi_ols.fit(X_m,y_m)                     # (특성,타겟) 모델훈련
multi = multi_ols.predict(X_m)             # 훈련된 모델로 X_m특성을 가진 타겟 예측
```
### 다항선형회귀(Polynomial Linear Reregrssion)
- 다항선형회귀는 2차항 이상의 독립변수로 종속변수를 설명하는 선형회귀
- 독립변수가 2개인 다항선형회귀(degree = 2)
- y = a1x2 + a2x2 + (a1x1**2) +(a2x2**2) + ax1x2 + b


### 회귀평가지표
* `MSE (Mean Squared Error)` = $\frac{1}{n}\sum_{i=1}^{n}(y_{i} - \hat{y_{i}})^{2}$
    - MSE는 제곱오차의 평균입니다.
    - MSE는 미분이 가능하여 비용 함수로 가장 흔하게 사용됩니다.
    - 하지만 제곱을 취하여 이상치에 민감합니다.
<br></br>

* `RMSE (Root Mean Squared Error)` = $\sqrt{MSE}$
    - RMSE는 MSE에 루트를 씌워 상대적으로 덜 민감하게 됩니다.
    - 값의 크기가 실제값과 비슷해지기 때문에 MSE에 비해 이해하기 쉽습니다.
<br></br>

* `MAE (Mean absolute error)` = $\frac{1}{n}\sum_{i=1}^{n}\left | y_{i} - \hat{y_{i}} \right |$
    - MAE는 절대오차의 평균으로 직관적으로 이해하기 쉽습니다.
    - 오차의 절대값이기 때문에 이상치에 민감하지 않습니다.
<br></br>

* `R-squared (Coefficient of determination)` = $1 - \frac{\sum_{i=1}^{n}(y_{i} - \hat{y_{i}})^{2}}{\sum_{i=1}^{n}(y_{i} - \bar{y_{i}})^{2}} = 1 - \frac{SSE}{SST} = \frac {SSR}{SST}$
    - $R^2$ (결정계수)는 모델의 설명력을 나타냅니다.
    - 값이 1에 가까울수록 설명력이 높으며 0에 가까울수록 설명력이 낮습니다.
<br></br>

- 참고
    - SSE(Sum of Squares `Error`, 관측치와 예측치 차이): $\sum_{i=1}^{n}(y_{i} - \hat{y_{i}})^{2}$
    - SSR(Sum of Squares due to `Regression`, 예측치와 평균 차이): $\sum_{i=1}^{n}(\hat{y_{i}} - \bar{y_{i}})^{2}$
    - SST(Sum of Squares `Total`, 관측치와 평균 차이): $\sum_{i=1}^{n}(y_{i} - \bar{y_{i}})^{2}$ , SSE + SSR
<br></br>
```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # r2 mae mse 임포트

def eval_models(y_pred, y_real=y_real) :         # 회귀평가지표 함수
    mse = mean_squared_error(y_real, y_pred)     # y_real : 실제값( 함수 앞에 미리 지정해둬야함)
    rmse = np.sqrt(mse)                          # y_red : 예측값
    mae = mean_absolute_error(y_real, y_pred)    # 함수에 예측값만 넣어주면 됨 
    r2 = r2_score(y_real, y_pred)

    return mse, rmse, mae, r2
```

### 선형회귀모델의 계수(Coefficients)
- 선형회귀모델의 장점은 직관적 해석이 가능하다는 점
- 독립변수와 종속변수 사이에 어떤 규칙을 학습했는지 알아볼 수 있다.
- .coef_ 회귀계수 , .intercept_ y절편
-  회귀계수 나오면 y = a1 * 회기계수1 + a2*회귀계수2 + y절편 으로 표현가능(a1 >a2라고 a1이 영향력이 큰건 아님(단순히 스케일의 차이일 수 있음)
```python
model.coef_         # 모델의 회귀계수 (다중이면 feature 순서대로 나옴)
model.intercept_    # 다중회귀 y절편
```

### 함수
```python
con = (df['GrLivArea']>=1700) & (df['GrLivArea']< 1800)     # 인덱스를 따로 설정하지 않아도 loc가능
df.loc[con, 'SalePrice'].min()  

sns.displot(df['SalePrice'], kde=True)   # 확률밀도함수

sns. regplot(x=,y=)                      # 회귀선 그래프 

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  
MSE (실제값, 예측값)     # 실제 타겟(target_feature, model)
RMSE (MSE)
R2(실제값, 예측값)

model.coef_              # 회귀계수  
model.intercept_         # y절편

```                            
