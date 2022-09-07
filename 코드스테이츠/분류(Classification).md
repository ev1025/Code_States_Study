# 분류(Classification)
1. 타겟 : 특정 범주에 속할 확률을 예측
   - 이진분류 : 타겟 값이 두 가지 범주(Yes or No)
   - 다중분류 : 타겟 값이 세 가지 이상인 범주(Class1, Class2, Class3)
2. 기준모델 : 타겟 변수의 최빈 클래스를 기준모델로 함
3. 분류모델 평가지표 (회귀모델 평가지표와 비교)
   - Accuracy (정확도) : (TP + TN) / (TP+TN+FP+FN)
   - Recall (재현율) : TP / (TP +FN)
   - Precision (정밀도)  : TP / (TP+FP)
   - F1 score : 2*precision*recall/(precision+recall)
   - roc_auc score
## Logistic Regression
- 선형회귀에 시그모이드(로지스틱)함수를 씌워서 확률을 구함
- 0과 1사이의 값을 출력하며 0.5 이상일 경우 1, 이하일 경우 0으로 분류
- skearn의 로지스틱회귀 모델은 디폴트로 L2 penalty(Ridge)가 적용되는 모델이라 표준화작업이 필요하다.
### 1. 결측치 ,중복치 삭제
### 2. 수치형데이터 아웃라이어 제거(boxplot)
### 3. 범주형데이터 인코딩(문자열인데 서열이 있는 자료들은 직접 순서 지정)
### 4. 모델링
#### 데이터셋 나누기(train, val, test)
```python
from sklearn.model_selection import train_test_split  # 데이터 나눠주기

X = df.drop('target',axis=1) # 특성 데이터
y = df['target']             # 타겟 데이터

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=2) # 훈련+검증데이터, 테스트데이터로 나눔(test_size = 테스트데이터 비율)
X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val, test_size=0.2, random_state=2) # 훈련데이터, 검증데이터로 나눔

print(X_train.shape, X_test.shape, X_val.shape)
```
#### 기준모델 생성
```python
base_major = y_train.mode()[0]            # 첫 번째 최빈값[0] 선택
y_pred_base = [base_major] * len(y_train) # 기준모델의 예측값
```
#### Scaling & Encoding
- Scaling
```python
from sklearn.preprocessing import StandardScaler            # 정규화
numeric_feats = X_train.dtypes[X_train.dtypes !="object".index # 숫자형 데이터 인덱스 출력
# numeric_feats = X_train[['age', 'height', 'weight', 'ap_hi', 'ap_lo']].columns # 데이터 타입이 제대로 안되어 있을 경우, 직접 수치 feats를 골라줘야함

scaler = StandardScaler()  # 각 train, val, test 데이터에 numeric_feats 인덱스값 넣어서 훈련, 예측값 생성
X_train[numeric_feats] = scaler.fit_transform(X_train[numeric_feats])
X_val[numeric_feats] = scaler.transform(X_val[numeric_feats])
X_test[numeric_feats] = scaler.transform(X_test[numeric_feats])
```
- Encoding
```python
!pip install category_encoders                   # 인코더 설치
from category_encoders import OneHotEncoder # 범주형데이터를 각 컬럼으로 변경

ohe = OneHotEncoder(use_cat_names = True, cols =['columns'])
# 데이터의 범주형 데이터를 수치형데이터로 변경해줌
# 데이터가 범주형이 아닐 때, cols로 지정하면 해당 컬럼을 변경
# use_cat_name =True 는 범주값이 열이름에 포함됨(false =인덱스포함, 디폴트)

X_train_ohe = ohe.fit_transform(X_train) # ohe 데이터 생성 (범주형-수치형)
X_val_ohe = ohe.transform(X_val)        
X_test_ohe = ohe.transform(X_test)      
```
#### 모델학습
```python
from sklearn.linear_model import LogisticRegression   # 로지스틱회귀
logistic = LogisticRegression(class_weight='balanced', max_iter=1000)
# max_iter = 경사하강 방법을 수행하는 횟수(수렴하면 성능이 더이상 늘지 않음) - solver가 Gradient Descent(경사하강)과 같이 weight(가중치)를 최적화하는 유형들을 구분 한 것
# class_weight = 클래스 가중치 설정

logistic.fit(X_train_ohe, y_train)         # 로지스틱 훈련
y_val_pred = logistic.predict(X_val_ohe) # 검증데이터 예측값 (필요한 데이터 예측하면 됨)

print('회귀계수가 양수인 특성 상위 3개')  # 절대값이 높을 수록 0이나 1에 가까움
print(pd.Series(logistic.coef_[0], logistic.feature_names_in_).sort_values(ascending=False).head(3), '\n')
print('회귀계수가 음수인 특성 하위 3개')
print(pd.Series(logistic.coef_[0], logistic.feature_names_in_).sort_values(ascending=False).tail(3))
```
#### 로지스틱데이터의 평가지표
```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
```
#### classfication_report
```python
from sklearn.metrics import classification_report
print(classification_report(y_val, y_val_pred))
>>>result # 평가지표 결과 모두 나옴
                precision    recall  f1-score   support

           0       0.71      0.77      0.74      5631
           1       0.75      0.68      0.71      5569

    accuracy                           0.73     11200
   macro avg       0.73      0.73      0.73     11200
weighted avg       0.73      0.73      0.73     11200
```
### 6. Counfusion Matrix
- TP(True Positive) : 실제 Positive인 것을 Positive라고 정확하게 예측한 경우
    - 심장병이 있는 샘플을 심장병이 있다고 예측한 경우
- FP(False Positive) : 실제로는 Negative인데 Positive라고 잘못 예측한 경우
    - 심장병이 없는 샘플을 심장병이 있다고 예측한 경우
- FN(False Negative) :  실제로는 Positive인데 Negative라고 잘못 예측한 경우
    - 심장병이 있는 샘플을 심장병이 없다고 예측한 경우
- TN(True Negative) : 실제 Negative인 것을 Negative라고 정확하게 예측한 경우
    - 심장병이 없는 샘플을 심장병이 없다고 예측한 경우
----
- 정확도(Accuracy)는 전체 범주를 모두 바르게 맞춘 경우를 전체 수로 나눈 값입니다: $\large \frac{TP + TN}{Total}$

- 정밀도(Precision)는 **Positive로 예측**한 경우 중 올바르게 Positive를 맞춘 비율입니다: $\large \frac{TP}{TP + FP}$

- 재현율(Recall, Sensitivity)은 **실제 Positive**인 것 중 올바르게 Positive를 맞춘 것의 비율 입니다: $\large \frac{TP}{TP + FN}$

- F1점수(F1 score)는 정밀도와 재현율의 조화평균(harmonic mean)입니다:  $ 2\cdot\large\frac{precision\cdot recall}{precision + recall}$
---
![캡처](https://user-images.githubusercontent.com/110000734/188880405-bd7a5669-83f1-4b28-bd92-00068dc32167.JPG)
- 행렬
```python
from sklearn.metrics import confusion_matrix # 행렬 생성
confusion_matrix(y_test, y_test_pred)[1][1]      # (00 = TN, 01 =  FP), (10= FN, 11=TP)
```
- 히트맵
```python
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
pcm = plot_confusion_matrix(logistic, X_test_ohe, y_test, # 모델, 특성, 타겟
                            cmap=plt.cm.Blues,
                            ax=ax)
plt.title(f'Confusion matrix of Logistic Regression, n = {len(y_val)}', fontsize=15)
plt.show();
```
### 7. Roc, Auc
```python
from sklearn.metrics import roc_auc_score

y_pred_proba = logist.predict_proba(X_test_ohe)[:,1] # 여러 행렬이 나옴??
auc = roc_auc_score(y_test, y_pred_proba)
```



### ETC
#### 임계값(thresholds)
- Recall과 Precision은 Trade-Off 관계
- 임계값을 낮추면 Recall이 증가 ( FN감소, FP 증가) 위양성증가
- 임계값을 높이면 Precision이 증가 (FN증가, FP 감소) 위음성증가
cs
릿지랑 랏쏘의 알파랑 같은 개념이라 생각하시면되어요. 규제의 강도를 설정하는 파라미터입니당
