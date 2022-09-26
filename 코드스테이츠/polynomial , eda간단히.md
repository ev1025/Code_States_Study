### np.polyfit(x, y, n) # 다항회귀, n = 차수 (1차시 출력값 :기울기, y절편) 
- 1차식 
```python
import numpy as np  # 다항회귀 구하기

x = [1, 2, 3, 4, 5, 6]
y = [1, 3, 6, 8, 11, 19]

print(np.polyfit(x, y, 1))
# [ 3.31428571 -3.6       ] -> y = 3.314x - 3.6
```

```python
import matplotlib.pyplot as plt # 다항회귀 시각화

poly = np.polyfit(x, y, 1) # 1차, 상수항 계수가 순서대로 들어있는 배열
y_pred = np.array(x) * poly[0] + poly[1] # x 값에 회귀를 진행한 예측 y 값

plt.plot(x, y_pred, color = 'b')
plt.scatter(x, y, color = 'r')
plt.show();
```
- 2차식
```python
print(np.polyfit(x, y, 2))
# [ 0.53571429 -0.43571429  1.4       ] -> y = 0.5357x^2 - 0.4357x + 1.4
```
```python
poly = np.polyfit(x, y, 2) # 2차, 1차, 상수항 계수가 순서대로 들어있는 배열
x_line = np.linspace(1, 6, 100) # 곡선을 완만하게 그리기 위한 linspace 객체 선언
y_pred = x_line ** 2 * poly[0] + x_line * poly[1] + poly[2]

plt.plot(x_line, y_pred, color = 'b')
plt.scatter(x, y, color = 'r')
plt.show()
```
### 이것저것 함수
```python
from google.colab import drive
drive.mount('/content/gdrive')


period = pd.date_range(start='2003-01-02', end='2022-09-23', freq='M')
# 월단위로 나누기 조정(마지막날로 묶임)

df["date2"].dt.strftime("%Y-%m-%d")
# 데이터타입을 바꿔주기
# df.index.strftime("%Y-%m-%d") dt를 사용하지않으면 DateIndex도 변경가능

replace(regex=True)
str와 numeric이 같이 있을 때, str.replace를 쓰면 numeric이 nan이 되어서 series.replace를 사용해야함 하지만 series.replace는 전체가 동일해야 작동하기때문에 regex=True 를 써줌

stocks = stocks.resample('M').mean()
# 월단위로 평균 값 묶어줌
```
