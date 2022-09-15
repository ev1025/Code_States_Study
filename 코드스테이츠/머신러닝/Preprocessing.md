# Preprocessing(전처리)

## Linear / Logistic Regression
1. 특성의 크기, 범위, 분포에 영향을 받음 -> 표준화, 스케일링 등 사용해야함
2. 수치적 연산이 학습에 이용되어 이상치, 결측치 -> 숫자
3. 비선형적인 특성이나 상호작용을 미리 처리해주어야함
  - log나 sqrt를 씌워서 타겟 특성과 선형일 것으로 예상되면 미리 전처리 해주어야함


## Tree Based Model
1. 입력 특성들의 크기, 범위, 분포에 영향을 받지 않음 -> 각 특성의 cut-line을 정해주는 학습하고 예측 
- 대소관계에 영향을 주지 않는 전처리는 영향을 주지 못함
2. 결측치를 채울 필요 없음(라이브러리에 따라 다름)
- 결측치를 하나의 집단으로 지정하기도 함
3. 특성과 타겟 간의 비선형적 관계나 특성의 상호작용이 자동으로 반영됨


## 결측치 
### 결측치 생성 이유
- 조건부 특성의 경우 특정 조건에서 발생(발병여부에서 여자에게만 걸리는병 -> 남자는 결측)
- 설문조사 특성 : 응답자가 의도적으로 비워둠(결측치 자체가 주는 시그널도 확인해야함)
- 엔지니어링 실수
### 결측치 처리방법
1. 그대로 두기 : 결측치에서 의미하는 바를 찾아낼 수 있음
- 선형회귀에서는 불가능하다.( 대신 특성들의 결측치를 나타내는 Bolean특성을 만들 수 있음)
- 트리기반 모델에서는 결측치를 핸들링 할 필요 없음(XGB에는 내장되어있고, sklearn에서는 결측치를 큰 값이나 작은값으로 채워넣어서 유사하게 다룰 수 있음)   

2. 단일 대표값으로 채우기
- 해당 데이터가 다른 특성들과 무관하게 발생했다는 가정(제일 많이사용) -> 데이터가 충분히 크거나 크기에 비해 결측치가 적을 때(~10% 내외) 사용
- sklearn의 simpleimputer(strategy=" ") : mean, median, most_frequent, constant(원하는값)

3. 다른 특성에서 빼와서 채우기
- 특성간의 관계나 정보를 바탕으로 결측치를 조건부로 채워줌
- 데이터 설명서를 꼼꼼히 확인하여 특정 특성의 결측이 다른 특성에 영향을 주는지 확인!
```python
interpolate(method=' ',                  # 'values' 선형비례하여 값 채우기(간격평균) /  ‘time'(날짜를 보고 채워줌) / nearest : 가까운값 / linear(두 값의 평균)
             limit = n,                  # limit = 채울 개수 제한
             limit_direction='backward') # 밑에서부터 채우기(default =foward) / both 중간부터 채우기?
             
fillna('인자’) # 결측값을 인자로 채우기 / value=dict 넣으면 해당 dict에서 각 열에 해당하는 값 넣어줌  
bfill() # 결측값을 바로 아래값과 동일하게 채워줌 / downcast = 'infer' float을 int로 바꾸어줌
ffill() # 결측값을 바로 위값과 동일하게 채워줌 
```
## 수치형 데이터 전처리
### 1. 각 분포의 스케일만 변환하는 모델들
#### MinMax scaling : 가장 단순하게 적용하는 기법   
  
$$ x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$    

- 0,1 사이로 스케일링 (사람의 나이정보, 딥러닝 에서 주로 사용)
- 이상치가 크지 않을 때, 특성값들의 분포가 균등분포(uniform distribution)일 때 사용

#### Standardization : 평균을0 표준편차를 1로 조정
  
$$ x' = \frac{x - \mu_{x}}{\sigma_{x}} $$ 

- $\mu_{x}$, $\sigma_{x}$은 각각 특성 $x$값들의 평균과 표준편차\
- Min-Max Scaling에 비해 이상치에 덜 민감하여 이상치 분포를 모를 때(또는 파악 이전에) 자주 사용
- 분포의 모양 자체는 변하지 않음

### 2. 값 분포의 스케일과 형태를 변화시키는 전처리
- 분포의 형태를 변화시킨다는 것은 각 값들 간 상대적인 거리도 변할 수 있다는 뜻
#### Clipping
- 특정 범위를 넘어서는 값들을 해당 경계값으로 변환시킴
  
$$ x' = min(max(x, x_{min}), x_{max}) $$ 

- $x_{min}$, $x_{max}$는 각각 우리가 설정해 주는 경계 값
- 정상 범위에서 값이 튀는 이상치가 있을 때 사용
- 분포모양은 변환시키지 않고, 이상치만 영향을 줌
> 사용방식 : df[‘columns'].clip(lower=0, upper=150)
#### 로그 변환(log1p, log(1+x))
  
$$ x' = log(1+x) $$ 

- 다수의 값들이 제한된 범위내에 존재하고 특정 값만 큰 형태를 보일 때 사용(파레토 법칙)
- 음수가 있을 경우 사용되지 않고, 로그에 0이 들어갈 수 없어서 1p or 1+x이라 함
> 사용방식 : np.log1p(df['columns'])
 
#### Bucketing
- 수치형, 범주형 데이터를 각각 범주화 하여 숫자로 나타내는 방법
- 범주화 방법
  - quantile : 범위 내에 동일한 수의 데이터가 들어가도록 쪼갬
  - uniform : 범위를 동일한 간격으로 쪼갬
  - 그 외에 도메인지식을 바탕으로 쪼갤 수 있음(과세표준 등..)
- 특정 특성이 중요하나, 타겟과 비선형일 때 범위를 잘 나누어 선형관계로 변환
```python
from sklearn.preprocessing import KBinsDiscretizer

# strategy = uniform / quantile / kmeans
# encode = ordinal / onehot
kbd = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="uniform")
```
#### Rank 반환
- 데이터를 전체 데이터의 순위로 변환(또는 percentile)
- 사용시 값들 간의 거리정보를 복구할 수 없는 대,소 관계만 남게 됨
- 이상치에 민감하지 않고 균등분포(Uniform Distribution)로 변환시켜주어 많이 사용
- pandas에 내장 된 .rank(pct=True: percentile 반환 / False: rank 자체를 반환)
- 기존에 없던 데이터가 들어오면 변환 불가(이럴 땐  sklearn의 QuantileTransformer 등을 사용)

## 범주형 데이터 전처리

### One-Hot Encoding / Ordinal Encoding
- `One-Hot Encoding`은 각 범주형 특성 값들을 각 값들에 대해 (해당한다/해당하지 않는다)의 0과 1 값으로 인코딩합니다. 
    - 명목형 변수는 One-Hot Encoding 방식을 이용해서 변환해줄 수 있습니다. 
    - 하지만 cardinarlity가 너무 클 경우에는 차원이 너무 커지기 때문에 적절한 인코딩 방법은 아닙니다. 
    - 트리 기반 모델에서는 특성의 정보를 분산시키고 비효율적은 트리를 만들 수 있기 때문에 잘 사용하지 않습니다.
- `Ordinal Encoding`은 각 범주형 특성 값들을 단순히 서로 다른 정수값들로 인코딩합니다. 
    - 순서형 변수는 Ordinal Encoding 방식을 이용해 변환해줄 수 있습니다. 
    - 선형 회귀나 로지스틱 회귀 모델을 사용할 때 명목형 변수를 ordinal encoding을 해주는 것은 적절하지 않습니다. 
        - 각 범주형 데이터가 양적 대소 관계를 갖는 것처럼 간주되기 때문입니다. 
    - 하지만 트리 기반 모델에서는 여러 번의 분기를 통해 이러한 양적 대소 관계가 점차 사라지기 때문에, `One-Hot Encoding`에 비해 효율적이면서 잘 작동합니다.

### Count Encoding(Frequency Encoding)
- 각 카테고리의 등장 빈도가 중요한 정보가 되는 경우(희귀 카테고리)
- 구현과 해석이 쉬우며 특성의 차원을 늘리지 않음
- 특성 빈도에 대한 정보를 모델에게 명시적으로 제공
- 동일한 빈도를 갖는 특성값이 구분되지 않음
- 새로 들어온 특성값에 대응이 어려움(미리 정해둔 값이나 결측치로 인코딩)
```python
from category_encoders import CountEncoder

count_encoder = CountEncoder(normalize=True)  # normalize=False -> frequency 자체를 반환합니다.
```
### Target Encoding(Mean Encoding)
- 범주형 특성값들을 해당 특성을 갖는 데이터의 타겟값 평균으로 인코딩
- 해석이 쉽고, 차원을 늘리지 않음
- 특성과 타겟 간의 직접적인 관계를 모델링하여 모델에게 중요한 정보 제공
- 서로 구분이 안됨, 과적합이 발생할 수 있음
- 인코딩 된 특성의 경우 값만 보아도 타겟값을 예측 할 수 있음
- 하지만, 학습데이터에 특성값이 충분히 포함되지 않은 경우 오해석될 수 있음
- 이에 라이브러리에 특성별 조건부 타겟평균, 해당출현빈도까지 고려하여 인코딩됨
```python
from category_encoders import TargetEncoder

target_encoder = TargetEncoder()
```


### 함수
np.random.normal(m, n, size) # 정규분포에서 무작위 표본 추출(평균, 표준편차, size)
np.random.uniform(low, high, size) # 균등분포로부터 무작위 표본 추출(최소값, 최대값 ,개수)
np.stack((a,b), axis=) # a와b의 행렬을 합침(a,b shape가 같아야하고 (2, 2)+(2, 2) ->(2, 2, 2)로 만듬

df = [1, 2, 3, 4]
reshape(2, 2)     # 데이터를 (2, 2) 형태로 만듬 / -1은 남는 만큼 채워줌
=>([[1, 2],
   [3, 4]])

{model.intercept_:.3f} + {model.coef_[0]:.3f}a + {model.coef_[1]:.3f}b" # 회귀식 쓰는거 : 절편, a회귀계수, b회귀계수
