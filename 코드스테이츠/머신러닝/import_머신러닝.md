# 머신러닝 지도학습 모델정리
### 선형회귀
```python
from sklearn.linear_model import LinearRegression # 선형회귀(단순, 다중)   

from sklearn.preprocessing import PolynomialFeatures # 다항 선형회귀
PolynomialFeatures(degree=)

from sklearn.linear.model import RidgeCV, LassoCV   
Ridge = RidgeCV(alphas=alphas, cv=n, random_state=) # alphas = np.arange(1,100,1)   
Lasso = LassoCV(alphas=alphas, cv=n, random_state=)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   # 회귀평가지표

model.coef_       # 회귀계수(값이 크다고 영향력이 큰 것은 아니다, 단위의 차이일 수 있음) <-> 분류의 feature_importance
model.intercept_  # y절편
# y = ax + b      # a = coef_  / b = intercept_
```
### 분류
```python
from sklearn.linear_model import LogisticRegression   # 선형회귀에 시그모이드 함수를 씌운 로지스틱회귀
from sklearn.linear_model import LogisticRegressionCV # 검증(cv)와 규제항(Cs)를 추가 할 수있는 로지스틱회귀

Cs = np.arange(1, 100, 1)

logistic_cv = LogisticRegressionCV(Cs=Cs, 
                                   cv=5, 
                                   max_iter=100,
                                   #class_weight = 'balanced’데이터가 불균형 할경우 데이터의 비율을 맞춰줌
                                   )
logistic_cv.C_  # 최적의 Cs값
logistic_cv.Cs_ # Cs의 목록


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report # 분류평가지표
from sklearn.metrics import roc_auc_score, roc_curve # 분류평가지표(0.5보다 작으면 0, 0.5보다 크면 1) 

y_proba = logistic.predict_proba(X_test_ohe)[:,1] # [:,0] = 0일 확률 ,[:,1] = 1일 확률
roc_auc_score(y_test, y_proba)                    # 1일 확률의 ROC점수
 
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
```
```python
from sklearn.metrics import plot_confusion_matrix         # 컨퓨전메트릭스
pcm = plot_confusion_matrix(logistic, X_test_ohe, y_test) # 예측값이 아닌 예측값 만들 데이터 넣어야함
pcm.confusion_matrix # plot아닌 행렬데이터
```
![캡처](https://user-images.githubusercontent.com/110000734/193766235-7b1c8c8f-611e-48a7-b39e-407554c16ef0.JPG)

```python
fpr, tpr, threshold = roc_curve(y_test, y_proba)                  # fpr, tpr, threshold 공식

roc = pd.DataFrame({'TPR':tpr, 'FPR':fpr, 'Threshold':threshold}) # fpr,tpr, threshold 데이터프레임

optimal_index = np.argmax(tpr-fpr)                                # tpr-fpr의 값이 최대인 인덱스
optimal_threshold = roc.Threshold[optimal_index]                  # 최대인 인덱스의 threshold

opt_fpr = roc[roc.Threshold == optimal_threshold]['FPR']          # 최대인 인덱스의 fpr
opt_tpr = roc[roc.Threshold == optimal_threshold]['TPR']          # 최대인 인덱스의 tpr

plt.plot(fpr, tpr)                                                # roc_curve
plt.scatter(opt_fpr, opt_tpr, c='red')                            # 최대인 인덱스의 threshold값
plt.plot(np.arange(0, 1.0, 0.01), np.arange(0, 1.0, 0.01),linestyle = '--')
plt.title('ROC_AUC_CURVE')
plt.xlabel('FPR')
plt.ylabel('TPR', rotation=90)
plt.show();
```
![image](https://user-images.githubusercontent.com/110000734/193826323-f26d32cb-b097-46a8-8cf0-b1605d322fbe.png)

### 데이터 검증
```python
from sklearn.model_selection import train_test_split   
train_test_split(test_size= , random_state=, Stratify=불균형비율데이터)   

from sklearn.model_selection import KFold 
kf = KFold(n_splits=n)   # n개로 분할   
kf.get_n_splits()        # n의 개수 출력   
kf.split(x_train)        # x_train을 cv돌림  

from sklearn.model_selection import cross_val_score   
cross_val_score(model, x, y, cv, scoring = "neg_mean_absolute_error") # 작을수록 좋은 평가지표는 neg 붙일 것(mae, mse, rmse)   

from sklearn.feature_selection import f_regression, SelectKBest   
selector = SelectKBest(score_func=f_regression, k=n)                  # 모델, transform으로 변환가능   
# n개의 특성만 사용하여 f_regression 방식으로 종속변수와 독립변수를 계산하는 방식   
selector.get_feature_names_out()                                      # 사용된 특성 목록(array형태)   
```
### 모델구축
```python
from sklearn.pipeline import make_pipeline                     # 메이크 파이프라인 

from sklearn.preprocessing import StandardScaler, MinMaxScaler # 표준화(분포 변화 x)

!pip install category_encoders                                 # 인코딩
from category_encoders import OneHotEncoder                    # 범주형특성의 0과 1인 특성 nunique()수 만큼 생성
ohe.category_mapping                                           # 인코딩 특성 확인   
```
