# 머신러닝 지도학습 모델정리
```python
from sklearn.linear_model import LinearRegression # 선형회귀(단순, 다중)   
from sklearn.linear.model import RidgeCV, LassoCV   
Ridge = RidgeCV(alphas=alphas, cv=n, random_state=) # alphas = np.arange(1,100,1)   
Lasso = LassoCV(alphas=alphas, cv=n, random_state=)   
from sklearn.preprocessing import PolynomialFeatures # 다항 선형회귀   

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   

from sklearn.model_selection import KFold   
kf = KFold(n_splits=n) # n개로 분할   
kf.get_n_splits()        # n의 개수 출력   
kf.split(x_train)         # x_train을 cv돌림  

from sklearn.model_selection import train_test_split   
train_test_split(test_size= , random_state=, Strategy=)   

from sklearn.model_selection import cross_val_score   
cross_val_score(model, x, y, cv, scoring = "neg_mean_absolute_error") # 작을수록 좋은 평가지표는 neg 붙일 것(mae, mse, rmse)   

from sklearn.feature_selection import f_regression, SelectKBest   
selector = SelectKBest(score_func=f_regression, k=n) # 모델, transform으로 변환가능   
# n개의 특성만 사용하여 f_regression 방식으로 종속변수와 독립변수를 계산하는 방식   
selector.get_feature_names_out() # 사용된 특성 목록(array형태)   

from sklearn.pipeline import make_pipeline # 메이크 파이프라인   


!pip install category_encoders   
from category_encoders import OneHotEncoder   
ohe.category_mapping # 인코딩 속성 확인   
```


