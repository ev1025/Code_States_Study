# 비지도학습 클러스터링 K-means
- 스케일링 된 자료에 k개의 군집을 지정하여 비슷한 값을 군집화하는 머신러닝 기법

## K-means
```python
from sklearn.preprocessing import StandardScaler  # 스케일링 불러오기
from sklearn.cluster import KMeans                # K-means 부러오기

scaler=StandardScaler()                           # 스케일러로 짧게 변경
df_sacled = scaler.fit_transform(df_sacler)       # 데이터 스케일링

sse = {}    필수아님                               #  Elbow Method 를 위한 데이터 생성

for k in range(1,10):   필수아님                   # 군집을 1개~9개까지 실험(군집 수 정해져 있으면 안해도됨)
  kmeans = KMeans(n_clusters=k,max_iter=n, random_state=1)   #  n_clusters=군집수, max_iter= 트레이닝수 , 1번에 고정
  kmeans.fit(df_scaled)                                      # 스케일링된 데이터를 트레이닝시킨다.(모델을 트레이닝)
  
sse[k] = kmeans.inertia_    필수아님              # Elbow Method의 y축, k = x축
                                                    Inertia는 각 클러스터 별 오차의 제곱의 합(분산)을 나타냅니다. 
                                                    각 데이터로부터 자신이 속한 군집의 중심까지의 거리를 의미합니다.
                                                    
cluster_labels = kmeans.labels_                   # 클러스터 값 저장
df2. = df_scaled.assign(Cluster = cluster_labels) # df2에 클러스터값이 포함된 스케일링데이터 지정

---------------------------------------------신뢰성 체크한 사례-----------------------------------------------------                 
label12 = pd.concat([label,label2], axis = 1)     # 라벨에 기본데이터 분류, 라벨2에 K-MEANS 결과 저장
label12.columns = ['label', 'label2']             
labelall = label12.query(' label == label2 ')     # 결과값과 기본 분류값이 같은 것 추출
acc = len(labelall)/len(label12)                  # 두 값이 일치하는 값 / 전체 값
```
#### ELBOW method 시각화
```python
plt.title('The Elbow Method') # 타이틀
plt.xlabel('Values of k')     # x축 이름
plt.ylabel('SSE')             # y축 이름
sns.pointplot(x=list(sse.keys()), y=list(sse.values())) # x축 = k , y축 = inertia
plt.show()
```
![68747470733a2f2f692e696d6775722e636f6d2f6474706175664a2e706e67](https://user-images.githubusercontent.com/110000734/185634991-10627b49-3034-4e56-98fd-a924926bf406.png)

### fit, train, fit_train 차이점
- 머신러닝 할 때, 훈련 데이터와 테스트 데이터를 나누어야하는데, 테스트 데이터를 테스트할 때는 transform로 테스트만 진행한다.
- fit(x) : 모델에 데이터를 넣어 모델의 연산에 따라 parameter를 업데이트하여 데이터에 적합한 모델을 구하는 것 (training) # 붕어빵 틀 만듬
- transform(x):  해당 모델이 가진 parameter 를 데이터에 적용하는 것 # 붕어빵 틀에 붕어빵을 만듬
- fit_transform(x) : 데이터 x에 대한 모델의 parameter를 구하고 이를 곧 x에 대해 적용하는 함수   


### 오늘의 함수 
```python
dropna ( axis = , how = all or any , subset=[‘ 컬럼1’,‘컬럼2’] )

np.where(조건식, 맞으면 x, 틀리면 y)
m = np.array([1,2,3])
n = np.where(x>1 , m , 0)            # n = [0, 2, 3] 1보다 큰값은 그대로 입력, 작은 값은 0으로 입력

str.find(“안녕”)                     # 안녕의 인덱스 찾기

df.dtype()                           # array의 데이터 타입 찾기

df.agg( 함수, axis = )               #  함수 결과 추출(‘sum’,‘min’, np.sum 여러개 써도 다 나옴)  columns
                                        ‘columns’ : ‘mean’  쓰면 컬럼명 밑에 mean값 추출    -->>  mean

np.log()                            # 로그값 계산 (반드시 실행 전 0값 제거!)

df = pd.to_datetime(df['columns'], format='%d-%m-%Y %H:%M')   # 날짜 데이터로 변경

total_seconds()                                          # 모든 시간을 초로 변경

df = df.drop(index = df[df["columns"] >= 25].index)      # 25보다 높거나 같은 값 제거

df.set_index( keys= [col] , inplace= , drop= )           # 해당 컬럼을 인덱스로 지정

np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # 넘파이 생략 없애기 threshold = 값개수 , linewidth = 문자 수

df3 = df2.assign(columns = '기존데이터')                  # 새 df3에 기존df2 +('기존데이터'에 컬럼이름지정) 지정
                                                           new_df3 = 기존df2.assign(new_columns_name = '기존 데이터'['수정/할당하려는 칼럼명'] +-*/(연산)
                                               
```
