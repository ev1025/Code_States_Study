# Pandas
```python
df.values      # 데이터를 넘파이형태로 출력

DataFrame(raw_data, columns = ['a' , 'b']  # a, b 컬럼만 추출
df.colunmnname / df['columnname']  # 해당 컬럼의 series 추출

del df['column'] # 컬럼 선택

df[["columns", "columns"]][:2] # 컬럼들의 2번쨰 행까지 꺼내기
df.loc[[row, row], ['column', 'column']] # 컬럼, 인덱스 열 꺼내기

df.drop(1) # index 1 제거 ([1,2,3,4]) 인덱스들 제거
df.drop('column', axis=1) # column 제거 ([‘column', 'column'], axis=1) 컬럼들 제거

s1 + s2  or s1.add(s2)  # Series 덧셈하면 같은 인덱스끼리 더해주고 한 쪽이라도 Nan이면 Nan으로 표시

df1.add(df2, fill_value=0) # DataFrame 덧셈에서 Nan값을 0으로 대체 후 계산 됨

Series + DataFrame # Series 값의 인덱스와 맞는 컬럼이나 인덱스에 맞춰서 모든 더해짐(broadcasting) / axis =0 , 1로 조정가능

df.replace( ['male','female'], [0, 1])  # male -> 0 / female -> 1 한 번에 변형
df.map({'male' : 0, 'female' : 1})
```

```python
f =lamda x : x.max() - x.min()  # apply series에 함수적용(통계치 구할 때 좋음)
df.apply(f)
>>>
column1   15313
column2   23153

--------------------------------------------------------------
def f(x):
    return Series([x.min(), x.max()], index = ["min", "max"])

df.apply(f)
>>>
# df의 min, max 데이터프레임 생성

------------------------------------------------------------
applymap(f) # 데이터프레임 변환에 사용(모든 데이터에 함수 적용)
```
```python
np.array(dict(enumerate(df['race'].unique()))) # unique값을 enumerate를 사용하여 (0, ’first')처럼 (인덱스,값) 형태로 나타내줌
----------------------------------------------------

value = list(map(int, np.array(list(enumerate(df['rate'].unique())))[:, 0].tolist()))
key = np.array(list(enumerate(df['race'].unique())), dtype=str)[:,1].tolist())

value, key # 출력(주로 값들 숫자로 인코딩 할 때 사용)
>>>
[0, 1, 2, 3], ['white', 'other', 'hispanic', 'black']
```

