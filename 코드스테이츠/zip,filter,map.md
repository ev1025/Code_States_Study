# 함수 이것저것 복습


train_test_split
```python
from sklearn.model_selection import train_test_split

train_test_split(df,
                 train_size=,           #  float(0~1) 둘이합쳐 1이어야함
                 test_size=,            #  float(0~1)
                 strategy = df[target], #  타겟의 비율이 불균형 할때 비율맞춰주는 함수
                 random_state=,
                 )
```
filter함수
- 조건에 맞는 데이터만 묶음
- list로 변환해야 추출가능
- list(filter(함수, iterable데이터))
```python
list(filter(lambda x: x[1] >= 0.015, feature_importances))
```
map함수
- 함수에 해당하는 값을 하나씩 계산해서 정렬(append처럼)
- list로 변환해야 추출가능
- list(map(함수, iterable데이터))
```python
 selected_feature_names = list(map(lambda x: x[0], features_selected))
```
zip 함수
- 같은 shape의 튜플이나 리스트를 묶어줌
- list로 변환해야 추출가능
- list(zip(튜플, 튜플)) or list(zip(list, list))
```python
(zip(X_train.columns, pipe.named_steps["xgbclassifier"].feature_importances_))
```
- feature_names_in_   : 모델의 특성 목록
- feature_importance_ : 모델의 특성중요도 추출
- named_steps['xgbclassifier'] : 파이프모델의 일부 꺼내서 사용
```python
pd. Series(logistic.coef_[0], logistric.feature_names_in_)

pipe.named_steps["xgbclassifier"].feature_importances_
# 파이프모델의 xgb값의 특성중요도 추출
```
피어슨(회귀) / 스피어만(분류) : 통계량기반 특성 선택
```python
# from scipy.stats import spearmanr

# np.corrcoef(x, y)[0, 1] # 두 특성  x, y의 피어슨 상관계수(2x2배열이라서 인덱스 사용)
# spearmanr(x, y)         # 두 특 성 x, y의 스피어만 상관계수
```
