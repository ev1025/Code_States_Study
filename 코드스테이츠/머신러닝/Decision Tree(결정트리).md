# Decision Tree(결정트리)
- 비선형 데이터의 패턴을 잘 학습하는 모델
- 비용함수를 최소로 하는 특성과 그 값에 대한 Yes/No Question의 대답으로 타겟 데이터를 분할하는 알고리즘
- 노드(node) : 질문이나 정답을 담고있는 결과
  - Root(뿌리)노드 : 분기시작 노드
  - internal(중간)노드 : 분기중간 노드(inner, branch)
  - leaf(잎)노드 : 분기가 끝난 노드(terminal, outer)
- 엣지(edge) : 노드를 연결하는 선
- 회귀, 분류 모두 적용가능
  - 회귀문제의 비용함수 : MSE (마지막노드의 타겟값들의 평균)
  - 분류문제의 비용함수 : 지니불순도 (마지막노드의 타겟값들의 최빈값), 엔트로피
### 결정트리의 장점
- 데이터 분할 과정이 직관적이고 시각화가 가능
- 데이터 전처리 과정이 많이 필요하지 않음(표준화, 더미변수 생성(인코딩), 결측치 처리)
- 분기하는 과정에서 특성 간 상호작용을 자연스럽게 포착
- 다중 출력 문제를 풀 수 있음
- 결정트리모델은 비선형, 비단조 데이터 분석에 용이
### 결정트리의 단점
- 훈련데이터의 제약이 없는 유연한 모델 -> 과적합 위험이 큼
- 작은 데이터 변동으로도 다른 트리 생성 -> 불안정
- 각 노드별 국소 최적의사결정이 이루어지는 탐욕알고리즘(greedy algoritm)을 기반으로 하여 전체 최적의사결정이라고 볼 수 없음
- 주어진 데이터를 분할하고 분할된 근사치를 예측값으로 반환하기 때문에 외삽이 어렵다.

## 결정트리모델
### 1. 데이터 확인
  - train, val, test 데이터로 나누어줌
```python
from sklearn.model_selection import train_test_split

train, val = train_test_split(train, test_size=0.2, stratify=train[target], random_state=2) # stratify를 타겟값으로 지정하면 0과 1의 비율을 맞춰줌(지정해주지 않으면, 모델의 성능차이 많이남)
train.shape, val.shape, test.shape
```
  - 타겟의 비율 확인 : train[target].value_counts(normalize=True) 
  - pandas_profiling을 사용하여 데이터 확인 후 EDA진행
```python
!pip install category_encoders
pip install --upgrade pandas-profiling          # 업데이트 후 런타임 재시작
```
```python
from pandas_profiling import ProfileReport   # EDA필요한 데이터들 찾아줌
profile = ProfileReport(train, minimal=True)   # minimal 변경하면 더 많은 분석 해줌
profile
```
### 2. 데이터 분리
- 각 train, val, test의 특성(x), 타겟(y)
```python
def xy_split(df):
    x = df.drop('vacc_h1n1_f', axis = 1)  # 타겟 데이터 제거
    y = df['vacc_h1n1_f']                 # 타겟 데이터 추출
    return x, y

x_train, y_train = xy_split(train)
x_val, y_val = xy_split(val)
x_test = test                              # 케글 제출 자료라 y_test 없음
```
- 기준모델 생성, 정확도 확인
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

base = train[target].mode()[0]
baseline = [base]*len(train)
base_acc = accuracy_score(y_train, baseline)
round(base_acc, 2) 
```
### 3. 파이프라인을 이용한 결정모델
- 편의성, 캡슐화(encapsulation) : fit과 predict 한 번에 전체 모델의 과정을 수행, 가독성이 좋다.
- 하이퍼파라미터 : 한 번에 모든 필요한 하이퍼파라미터를 서치할 수 있다.
- 안전성 : test data의 정보가 train data로 누수되는 것을 방지할 수 있다.
```python
from sklearn.pipeline import make_pipeline       # 파이프라인 만들기
from sklearn.impute import SimpleImputer         # 결측값 채우기 - > (strategy : default='mean')
from category_encoders import OrdinalEncoder     # Ordinal 인코딩 -> 서열형 데이터 인코딩(1부터 시작)
from sklearn.tree import DecisionTreeClassifier  # 결정트리모델 

pipe_dt =  make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(),
    DecisionTreeClassifier(random_state=2, 
                           criterion = "entropy",  # criterion = 디폴트:지니불순도("gini"), "entropy"
                           max_depth = 6)          # max_depth = 분기 개수
)
pipe_dt.fit(x_train, y_train)                                      # 인코딩, 결측값제거, 결정트리모델 파이프라인으로 한 번에

print(pipe_dt.score(x_train, y_train), pipe_dt.score(x_val,y_val)) # train, val의 정확도(accuracy) -.score로 가능

y_pred = pipe_dt.predict(x_test)                                   # test데이터의 예측값
```
### 4. 특성중요도
- 결정트리모델의 해석 방법 : 각 특성이 얼마나 먼저, 자주 분기에 사용되었는지에 따라 특성중요도를 계산.
- 특성중요도 시각화
```python
%matplotlib inline             
import matplotlib.pyplot as plt   # 특성중요도 그래프

pipe_dt = pipe_dt.named_steps['decisiontreeclassifier']
importances = pd.Series(pipe_dt.feature_importances_, x_train.columns)
plt.figure(figsize=(10, 20))
importances.sort_values().plot.barh();
```
- 결정트리모델 시각화
```python
import graphviz
from sklearn.tree import export_graphviz

model_dt = pipe_dt.named_steps['decisiontreeclassifier'] # named_steps 속성을 사용해서 파이프라인의 각 스텝에 접근이 가능합니다.
enc = pipe_dt.named_steps['ordinalencoder'] # named_steps 은 유사 딕셔너리 객체(dictionary-like object)로 파이프라인 내 과정에 접근 가능하도록 합니다.
encoded_columns = enc.transform(X_val).columns

dot_data = export_graphviz(model_dt
                          , max_depth=3
                          , feature_names=encoded_columns
                          , class_names=['no', 'yes']
                          , filled=True
                          , proportion=True)


display(graphviz.Source(dot_data))
```
- 결정트리모델 과적합 해소방법 : 하이퍼파라미터를 사용하여 복잡도를 줄인다.
- min_samples_split / min_samples_leaf / max_depth = None(끝까지 분기)

## 랜덤포레스트(RandomForest)
- 결정트리를 기본모델로 사용하는 앙상블 기법
- 배깅(bagging)을 활용하여 결정트리모델의 과적합과 불안정성 해소
- 기본 트리들은 배깅을 통해 만들어지며, 각 기본 모델에 사용되는 데이터는 랜덤
- 노드를 분할 할 때 n개의 특성 중 일부 k개의 특성을 랜덤하게 선택하여 최적특성을 찾아냄
- 이 때 k개는 일반적으로 log2n을 사용함 

### 배깅(bagging, Bootstrap + AGGregatING)
- Train 데이터에서 무작위로 복원추출(중복허용추출)한 샘플로 여러 개의 개별 모델을 만든 뒤 예측 결과를 종합하여 최종 예측을 하는 방법
- 예측하는 과정에서 랜덤성을 부여하고 분산을 줄여 과적합을 해소
### Out-of Bag samples(oob sample)
- oob sample은 부트스트랩에서 한 번도 추출되지 않은 샘플을 의미한다.
- 데이터가 충분히 크다고 가정했을 때,   
> 한 부트스트랩 세트는 표본의 63.2%에 해당하는 샘플을 가짐
> 여기서 추출되지 않은 36.8%의 샘플이 Out-Of-Bag 샘플이며 이것으로 데이터검증
### Aggregation
- 부트스트랩세트로 만들어진 기본모델들을 합치는 과정
- 회귀문제 : 결과의 평균
- 분류문제 : 다수결로 가장 많은 모델이 선택한 범주
### 랜덤포레스트
```python
from sklearn.ensemble import RandomForestClassifier

pipe_random = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(),
    RandomForestClassifier(random_state=2,
                           min_samples_leaf = 5,  # min_sample_leaf 분기 최소 샘플수  
                           max_depth=20,          # 밑의 3개는 랜덤 포레스트에서만 사용가능
                           max_features = 30,     # max_features : 분할에 사용되는 최대 특성의 수(최대 특성의 수까지)
                           n_estimators =200,     # n_estimators : 기본 모델의 수
                           oob_score=True)        # oob_score : oob sample을 이용한 검증 스코어 반환 여부
)
pipe_random.fit(x_train, y_train)
print(f"훈련 : {pipe_random.score(x_train, y_train)}")
print(f"검증 : {pipe_random.score(x_val,y_val)}")
```
#### oob_score
```python
# out-of-bag sample을 이용하여 oob_score를 구할 수 있습니다.
pipe_random.named_steps["randomforestclassifier"].oob_score_
```
#### 특성중요도
```python
%matplotlib inline
import matplotlib.pyplot as plt

model_rf = pipe_random.named_steps['randomforestclassifier']
importances = pd.Series(model_rf.feature_importances_, x_train.columns)
plt.figure(figsize=(10, 20))
importances.sort_values().plot.barh();
```



#### 무엇인가??
min_sample_split ?
.named_steps ? 
.feature_importances_ ?
