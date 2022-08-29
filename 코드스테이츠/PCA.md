# PCA   

#### 공분산(Covariance)   
- 두변수에 대하여 한 변수가 변화할 때 다른 변수가 어떤 연관성을 갖고 변하는지 나타낸 값
- df.cov() or np.cov()            # 공분산

#### 분산-공분산행렬(variance-covariance matrix)   
- 모든 변수에 대하여 분산과 공분산 값을 나타내는 정사각 행렬
- 주대각선은 자신의 분산을 나타냄   

#### 상관계수(Correlation coefficient)
- 공분산을 두 변수의 표준편차로 나눠준 값
- 공분산의 스케일을 조정하는 효과
- 변수의 스케일에 영향을 받지 않음
- -1에서 1사이의 값을 가짐(1 = 변수들이 완벽한 양의 상관관계)
-  df.corr() or np.corrcoef()    # 상관계수   
#### Vector Transformation
- 임의의 벡터에 행렬T를 곱하여 변화시키는 것 (np.matmul) @(행렬곱)
- 일반적으로 단위벡터
-  Eigenvector : Transformation에 의해서 크기만 변하고 방향은 변하지 않는 것(여러가지)
-  Eigenvalue : eigenvector의 변화값(고유한 값)
-  Eigenstuff : np.linalg.eig() # Eigenvalue, Eigenvector 형식으로 나옴, T행렬의 컬럼수만큼 생성

## 주성분분석(Principal Component Analysis, PCA)   
- 원래 데이터의 정보(분산)을 최대한 보존하는 새로운 축을 찾고, 그 축에 데이터를 사영하여 고차원의 데이터를 저차원으로 변환하는 기법
- 주성분(PC)는 기존 데이터의 분산을 최대한 보존하도록 데이터를 projection하는 축
- 주성분(PC)의 단위벡터는 데이터의 공분산 행렬에 대한 eigenvector
- eigenverctor에 사영한 데이터의 분산이 eigenvalue

### 방법   
1. 데이터 표준화 : 각 열에 대해 평균을 빼고 표준편차로 나누어 데이터를 평균0 표준편차1로 scaling 하는 것
2. 표준화 한 데이터 셋의 공분산 구하기
3. 공분산의 eigen stuff 구하기( 최대한 분산이 넓은 값)
4. eigenvector에 사영(projection)하기 (표준화 자료와 공분산의 아이겐 백터 내적을 구함) - PC값
</br>

#### PCA 수작업 PC 구하기
```python
# 1. 데이터 표준화(0~1 로 만드는 과정, 표준편차가 1)
standardized_data = ( features - np.mean(features, axis = 0) ) / np.std(features, ddof=1, axis = 0)
print("\n Standardized Data: \n", standardized_data)
standardized_data = np.array(standardized_data)

# 2. 표준화된 공분산(주대각선이 1, 주대각선이 본인의 분산인데, 표준화과정에서 표준편차를 1로 맞췄으니 분산도1^2 해서 1이됨)
covariance_matrix = np.cov(standardized_data.T)
print("\n Covariance Matrix: \n", covariance_matrix)

# 3. 공분산의 eigen stuff(vector,value) 구하기(최대한 넓은 분산)
values, vectors = np.linalg.eig(covariance_matrix)
print("\n Eigenvalues: \n", values)
print("\n Eigenvectors: \n", vectors)

# 4. eigenvector에 projection(사영)  - 표준화 데이터와 공분산의 eigenvector 내적 = PC값
Z = np.matmul(standardized_data, vectors)

print("\n Projected Data(Z): \n")
pd.DataFrame(Z, columns=['pc1', 'pc2','pc3','pc4'])
```
</br>

#### PCA함수 사용하여 PC 구하기
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Data: \n", features)

# 1. StandardScaler()를 사용하여 데이터 표준화 
scaler = StandardScaler()
Z = scaler.fit_transform(features) # fit_transform은 학습데이터에 사용할때 씀<>transfom 은 test 데이터에 사용
print("\n Standardized Data: \n", Z)

# 2. 표준화한 데이터에 대하여 pca 시행 
pca = PCA(2) # PC 갯수 설정
pca.fit(Z)   # 주성분 찾기(projection)

print("\n Eigenvectors: \n", pca.components_) # 공분산의 Eigen vector
print("\n Eigenvalues: \n",pca.explained_variance_) # 공분산의 Eigen value

B = pca.transform(Z) # 새로운 주성분으로 데이터 변환 (표준화데이터와 공분산의 내적을 곱하여 PC 생성 )
print("\n Projected Data: \n", B)

pca.explained_variance_ratio_  # 전체 변동성에서 개별 PCA component 별 차지하는 변동성 비율(설명력)
```
### PCA 시각화
```python
B = pd.DataFrame(B, columns=['pc1', 'pc2'])     # 데이터프레임 만듬(넘파이 쓰기 힘들어서)
sns.scatterplot(data=B, x='pc1', y='pc2', hue='species') 
plt.legend(bbox_to_anchor=(1.01, 1.02))
plt.show()
```




### 오늘의 함수

```python
df.avr() or np.var(df, ddof=1)  # 분산
df.cov() or np.cov()            # 공분산
df.corr() or np.corrcoef()      # 상관계수

np.linalg.eig()                 # eigen stuff 구하는 공식

np.around(x, n)                 # 넘파이에 사용하는 반올림..

import seaborn as sns
df = sns.load_dataset('penguins')  # 펭귄샘플데이터 불러오기

import pandas as pd
series.to_frame()                  # 시리즈를 데이터프레임으로 변경

변수 = pd.DataFrame(array변수)      #array를 dataframe형식으로

df =  df.drop([[df.index[3],df.index[339]])  # df의 인덱스값 제거

matmul 대신 a@b = a행렬과 b행렬의 행렬곱
```
