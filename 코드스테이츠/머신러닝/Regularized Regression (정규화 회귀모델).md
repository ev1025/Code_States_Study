# Regularized Regression (정규화 회귀모델)

### 과적합을 줄이는 방법
- 데이터의 사이즈에 따라 적절한 수준의 복잡도를 선택해야한다. 
- 너무 복잡한 모델은 노이즈까지 학습하기 때문에, 모델의 복잡도를 줄인다.
- 데이터가 부족한 상태에서의 과적합은 더 많은 데이터를 학습시켜 적절한 수준의 복잡도를 가진 모델로 만든다.
- 모델에 규제항(lambda)를 더해 기존 모델보다 단순하게 만든다.-> 회귀계수를 축소하여 과적합 예방

### 정규화 회귀모델
- 정규화 회귀모델 : 선형모델에 규제항(lambda)을 더해 과적합을 방지(선형모델이 학습데이터에 덜 적합하게 만들어 일반화)
- 정규화란 모델에 편향을 올리고 분산을 줄여서 일반화 (분산, 편향 Trade off)
- 정규화 회귀모델은 입력 특성의 스케일에 민감하여 반드시 스케일링(Standard Scalering) 작업이 필요
- 규제항의 종류
  - Lasso : L1 Penalty(|가중치| 절대값의 합)
  - Ridge : L2 Penalty(np.power(가중치) 제곱의 합)
  - ElasticNet : L1 + L2

### Lasso Regression
- 회귀계수에 가중치 절대값의 합(L1 Penalty)를 패널티로 부과하여 회귀계수 축소
- 영향력이 크지않은 회귀계수 값을 0으로 만들어 영향력 있는 변수만 선택 가능
- 자동으로 특성을 선택하는 효과를 가지게 되며 가중치가 0인 특성이 많은 희소모델(sparse model)을 만들게 됨
- 적은 수의 설명변수가 큰 회귀계수를 가질 때 좋은 예측 가능
- 일부의 설명변수만 포함하므로 단순하고 해석력 높은 모델을 만들 수 있음
- 변수간 상관관계가 높은 상황에서 오히려 예측력이 떨어짐

$$ cost = RSS + \lambda\sum_{j=1}^p\vert\beta_j\vert = \sum_{i=1}^n(y_i - \beta_0 - \beta_1x_{i1}-\dotsc-\beta_px_{ip})^2 + \lambda\sum_{j=1}^p\vert\beta_j\vert $$

### Ridge Regression
- 회귀계수에 가중치 제곱합(L2 Penalty)을 패널티로 부과하여 회귀계수 축소
- 영향력이 크지 않은 값은 0에 가깝게 수렴 -> 모든 특성을 사용
- 변수 간 상관관계가 높은 상황(회귀계수의 크기가 비슷할 때)에서 좋은 예측 가능
- 미분이 가능해 Gradient Descent 최적화 가능
$$ cost = RSS + \lambda\sum_{j=1}^p\beta_j^2 = \sum_{i=1}^n(y_i - \beta_0 - \beta_1x_{i1}-\dotsc-\beta_px_{ip})^2 + \lambda\sum_{j=1}^p\beta_j^2 $$

### $\lambda$ (Lambda)
- $\lambda$ 는 패널티의 강도를 조절하는 하이퍼파라미터
- lambda, alpha, penalty term, regularization term 등으로 불림
- $\lambda$ 의 크기가 클수록 회귀계수의 값이 줄어듬
    - $\lambda = 0$ 인 경우 기존의 선형회귀와 같음
    - $\lambda = \infty $ 인 경우  $\beta = 0$ 이 됩니다.

### Lasso Regression
```python
lasso = Lasso(alpha=0.1, normalize=True)     # alpha : 패널티, normalize : 표준화 작업
lasso.fit(ans[['x']], ans['y'])
y_pred_lasso = lasso.predict(ans[['x']])

ans.plot.scatter('x', 'y')                   # 그래프 그리기
plt.plot(ans['x'], y_pred_lasso)
plt.title(f'y= {lasso.coef_[0].round(2)}x + {lasso.intercept_.round(2)}') #coef_ 기울기 , intercept y절편
plt.show();
```
### Ridge regression
```python
ridge = Ridge(alpha=0.5, normalize=True) # alpha : 패널티, normalize : 표준화 작업
ridge.fit(ans[['x']], ans['y'])                # 모델 훈련
y_pred_ridge = ridge.predict(ans[['x']])       # 예측값 생성

ans.plot.scatter('x', 'y')                     # 그래프
plt.plot(ans['x'], y_pred_ridge)
plt.title(f'y= {ridge.coef_[0].round(2)}x + {ridge.intercept_.round(2)}') #coef_ 기울기 , intercept y절편
plt.show();
```
### 다항선형회귀의 회귀계수
```python
# ols 회귀계수
ols_coef = ols.named_steps['linearregression'].coef_

# Ridge 회귀계수
def PolynomialRidge(degree=10, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), 
                         Ridge(**kwargs))
ridge = PolynomialRidge(alpha = 0.0000001, normalize=True)
ridge.fit(ans[['x']], ans['y'])
ridge_coef = ridge.named_steps['ridge'].coef_

# Lasso 회귀계수
def PolynomialLasso(degree=10, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), 
                         Lasso(**kwargs))

lasso = PolynomialLasso(alpha = 0.1, normalize=True)
lasso.fit(ans[['x']], ans['y'])
lasso_coef = lasso.named_steps['lasso'].coef_

coef_comparison = pd.DataFrame(index = [1, 'x', 'x^2', 'x^3', 'x^4', 'x^5', 'x^6', 'x^7', 'x^8', 'x^9', 'x^10'], 
                            data = {'OLS':ols_coef,'Ridge':ridge_coef, 'Lasso' :lasso_coef}) # 표 생성
```
## Case study
### 전처리
- 결측치, 중복값, 고유값 삭제
```python
df.duplicated().sum()                              # 중복되는 값이 있는 확인

df.isna().sum()[df.isna().sum() !=0]/len(df)       # 결측치를 확인.

cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] # 결측치가 80%가 넘는 컬럼도 삭제
df.drop(cols, axis=1, inplace=True)


df.nunique()[df.nunique()/len(df)>0.7]             # 고유값 많은 컬럼 삭제
cols = ['Id', 'LotArea']
df.drop(cols, axis=1, inplace=True)
```
### 범주형, 수치형 컬럼 수치화 하는 방법
```python
상관관계가 높은 컬럼 중 범주형 컬럼과 타겟의 관계 및 수치형 컬럼과 타겟의 관계를 시각화

범주형 컬럼 
1. OverallQual : 전반적인 퀄리티 (1-10)
2. GarageCars : 주차장에 주차할 수 있는 차의 수(0-4)
3. FullBath : 욕실의 수(0-3)

수치형 컬럼
1. GrLivArea : 집의 크기
2. GarageArea : 주차장의 크기
3. TotalBsmtSF : 지하실의 크기

# 범주형 컬럼과 타겟과의 관계
fig = plt.figure()
fig.set_size_inches(24, 12)
(ax1, ax2, ax3), (ax4, ax5, ax6) = fig.subplots(nrows=2, ncols=3)
sns.boxplot(data=df, x='OverallQual', y='SalePrice', ax=ax1)
ax1.set_title('SalePrice - OverallQual')
sns.boxplot(data=df, x='GarageCars', y='SalePrice', ax=ax2)
ax2.set_title('SalePrice - GarageCars')
sns.boxplot(data=df, x='FullBath', y='SalePrice', ax=ax3)
ax3.set_title('SalePrice - FullBath')

# 수치형 컬럼과 타겟과의 관계
sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', ax=ax4)
ax4.set_title('SalePrice - OverallQual')
sns.scatterplot(data=df, x='GarageArea', y='SalePrice', ax=ax5)
ax5.set_title('SalePrice - GarageArea')
sns.scatterplot(data=df, x='TotalBsmtSF', y='SalePrice', ax=ax6)
ax6.set_title('SalePrice - TotalBsmtSF')
plt.show()
```
### 모델링
```python
from sklearn.model_selection import train_test_split
X = df.drop('SalePrice', axis=1)   # 특성추출
y = df['SalePrice']                # 타겟추출 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 특성, 타겟 훈련용, 테스트용으로 나눔
```
#### Preprocessing : Scaling & Encoding
- 결측치를 먼저 평균으로 모두 채워주겠습니다. 
```python
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
```
1. Scaling : StandardSacler로 표준화
```python
numeric_feats = X_train.dtypes[X.train.dtypes != "object"].index      # 수치형데이터 인덱스 추출

scaler = StandardScaler()
X_train[numeric_feats] = scaler.fit_transform(X_train[numeric_feats]) # X_train의 수치형 데이터 스케일링 , 평균0 표준편차 1
X_test[numeric_feats] = scaler.tranform(X_test[numeric_feats])        # X_test의 수치형데이터 스케일링 , 평균0 표준편차 1
 
X_train[numeric_feats].describe().T[['mean', 'std']] # 평균, 표준편차 확인
```
2. One -Hot encoding
- 문자형 데이터를 숫자형 데이터로 변경
- 각 인자를 컬럼으로 변경하여 0 1 로 표시
- 범주가 많은 경우 적절하지 않음
```python
!pip install category_encoders              # 핫인코딩 라이브러리 설치

from category_encoders import OneHotEncoder
ohe = OneHotEncoder()

X_train_ohe = ohe.fit_transform(X_train)    # 문자형 -> 목록형 스케일링
X_test_ohe = ohe.transform(X_test)       

(X_train_ohe.dtypes == 'object').sum()      # 문자형 데이터가 없는 것을 확인
```
### 기준모델
```python
from sklearn.metrics import r2_score, mean_absolute_error

baseline = [y_train.mean()] * len(y_train)
baseline_r2 = r2_score(y_train, baseline)
baseline_mae = mean_absolute_error(y_train, baseline)
print(f'기준모델의 r2_score: {baseline_r2}')
print(f'기준모델의 mae : {baseline_mae}')
```
### 다중선형회귀(OLS)
```python
def print_score(model, X_train, y_train, X_test, y_test) :    # R2 구하는 함수

    train_score = np.round(model.score(X_train, y_train) , 3)
    val_score = np.round(np.mean(cross_val_score(model, X_train, y_train, scoring='r2', cv=3).round(3)),3)
    test_score = np.round(model.score(X_test, y_test),3)
    print(f'학습 세트 r2_score : {train_score}')
    print(f'검증 세트 r2_score : {val_score}')
    print(f'테스트 세트 r2_score : {test_score}')

    return train_score, val_score, test_score

from sklearn.model_selection import cross_val_score # 교차검증 불러오기

ols = LinearRegression()            
ols.fit(X_train_ohe, y_train)                       # 모델훈련

ols_train, ols_val, ols_test = print_score(ols, X_train_ohe, y_train, X_test_ohe, y_test) # 모델, 특성,타겟
```
### feature selection
- 과적합이 발생하였을 때 일부의 특성만 사용하는 것(복잡도 줄이기)
```python
from sklearn.feature_selection import f_regression, SelectKBest
selector = SelectKBest(score_func= f_regression, k=50)          # 50개의 특성만 추출

x_train_selected = selector.fit_transform(x_train_ohe, y_train) # 전처리 데이터 중  50개의 특성만 훈련
x_test_selected = selector.transform(x_train_ohe)               # 테스트데이터 예측값

selector.get_feature_names_out()                                # 50개의 feature 확인

ols_fs = LinearRegression()
ols_fs.fit(X_train_selected, y_train)                           # 50개 다중 선형회귀

ols_fs_train, ols_fs_val, ols_fs_test = print_score(ols_fs, X_train_selected, y_train, X_test_selected, y_test)
```
### Ridge regression
```python
for alpha in [0.0.1, 0,1 1,0, 1, 100.0, 1000.0, 10000,0]

    print(f'Ridge Regression, alpha={alpha}')

    ridge = Ridge(alpha=alpha)      # Ridge 모델 학습
    ridge.fit(X_train_ohe, y_train)

    print_score(ridge, X_train_ohe, y_train, X_test_ohe, y_test)   # 성능 확인(위에 함수)
    coefficients = pd.Series(ridge.coef_, X_train_ohe.columns)     # plot coefficients
    idx = np.abs(coefficients).head(40).index                      # 절대값 상위 40개의 회귀계수
    plt.figure(figsize=(6, 8))
    coefficients[idx].sort_values().plot.barh()
    plt.show()
```

### Lasso regression
```python
for alpha in [0.000001, 0.001, 0.01, 1.0, 100]: 
        
    print(f'Lasso Regression, alpha={alpha}')

    lasso = Lasso(alpha=alpha)  
    lasso.fit(X_train_ohe, y_train)
x
    print_score(lasso, X_train_ohe, y_train, X_test_ohe, y_test) # 성능 확인


    coefficients = pd.Series(lasso.coef_, X_train_ohe.columns)
    idx = np.abs(coefficients).head(40).index                    # 상위 40개 그래프
    plt.figure(figsize=(6, 8))
    coefficients[idx].sort_values().plot.barh()
    plt.show()

```

### LassoCV, RidgeCV
-최적의 alpha값을 찾는 방법
```python
from sklearn.linear_model import RidgeCV, LassoCV  # CV 출력

alphas = np.arange(1, 100, 10)
ridge = RidgeCV(alphas=alphas, cv=5)               # alpha, cv값 지정
ridge.fit(X_train_ohe, y_train)                    # 모델 훈련

print("alpha: ", ridge.alpha_)                     # 최적 알파값

# 성능 확인
ridge_train, ridge_val, ridge_test = print_score(ridge, X_train_ohe, y_train, X_test_ohe, y_test)
```

### 최종 선정된 Lasso 모델
```python
X_total = pd.concat([X_train_ohe, X_test_ohe])   # 테스트, 검증데이터 합침(cv가 알아서 해줌)
y_total = pd.concat([y_train, y_test])


alphas = np.arange(10, 200, 10)

lasso_final = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_final.fit(X_total, y_total)
lasso_final_coef = lasso_final.coef_

print(f'alpha: {lasso_final.alpha_}')                            # 최종 알파값
print(f'cv best score: {lasso_final.score(X_total, y_total)}')   # 최종 r값

# 회귀계수를 확인 
print(f'lasso 회귀계수 최대값 : {lasso_final_coef.max()}\nlasso 회귀계수 평균 : {lasso_final_coef.mean()}')
print(f'회귀계수가 0이 아닌 특성의 수 : {(lasso_final_coef!=0).sum()}')
lasso_final_coef.sort()
plt.plot(lasso_final_coef)
plt.show()
```
