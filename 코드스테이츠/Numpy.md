# Numpy(Numerical Python)
- list에 비해서 메모리 효율적
- 반복문 없이 데이터 배열에 대한 처리를 지원함
- 선형대수와 관련된 다양한 기능을 제공
- C언어와 연계가능
- for문 절대 쓰지말것!!(손해 where이나 argmax, argmin 쓸것)
- 중요 한 것 : where / broadcasting / argmax / argmin / fancy / boolean index
```python
test_array = np.ararry(["1", "2", 3, 4], float)
>> result array ([1., 2., 3., 4.,])          # str로 입력해도 같은 타입으로만 저장됨
```
### Array shape, dmin
```python
np.array([1,2,3]).shape # 3열
(3,.) 
np.array([[1,2,3], [4,5,6]]).shape #(2행, 3열) shape 값이 하나씩 뒤로 밀림
(2, 3)
np.array([[1,2,3], [4,5,6]]).ndim # 차원(행렬의 행개수랑 같음)
2 
np.array([[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]]).shape # (2겹, 2차원(행), 3열)
(2, 2, 3)
```
### reshape
- array의 shape의 크기를 변경(element의 개수는 동일)
- 데이터의 차원과 상관없이 데이터 개수만 맞추면 가능
```python
np.array([[1,2,3,4], [1,2,3,4]]).reshape(2, 2, 2)   # (2, 4)
=> np.array([[[1,2], [3,4]], [[1,2], [3,4]]])
np.array([[1,2,3,4]], [[1,2,3,4]]).reshape(8,) # (2,4)
=>np.array([1,2,3,4,1,2,3,4]).shape       

np.array([1,2,3,4,1,2,3,4]).reshape(-1, 1) # (8,) -1은 남은 개수만큼 채워주라는 뜻
=> np.array([1],                         # (8, 1)
            [2],
            [3],
            [4],
            [1],
            [2],
            [3],
            [4])
```

### flatten
- 다차원 array를 1차원 array로 변경   
- np.array([[[1,2], [3,4]], [[1,2], [3,4]]]).flatten()   
=> np.array([1,2,3,4,1,2,3,4])

### arange
```python
np.arange(5)
=> np.array([0,1,2,3,4])

np.arange(0, 5, 0.5) # (시작, 끝, 스탭)
=> array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5)]) 

array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5)]).tolist() # LIST에서는 INT는 이런 배열 못만듬
=> [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)]
```
### Zeros & Ones
```python
np.zeros(shape=(6,), dtype=np.int8) # np.ones = 1로 채우는것
=> np.array([0, 0, 0, 0, 0, 0])

np.zeros((2,3))
=> np.array([[0, 0, 0], [0, 0, 0]])

# someting_like : 기존 ndarray의 shape만큼 1, 0 또는 empty array 반환
test = np.zeros((2,3))
np.ones_like(test)
=> np.array([[1, 1, 1], [1, 1, 1]])

# identity : 단위행렬 생성
np.identity(n=3, dtype=np.int8) # np.identity(3)
=>array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int8)

# eye : 단위행렬 생성
np.eye(N=3, M=5, dtype=np.int8) # N = row / M= columns / K = 단위행렬 시작열
=> array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=int8)

# diag(다이고날) : 대각행렬의 값을 추출
a = np.arange(9).reshape(3,3)
=> 
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
np.diag(a)         # 대각선 추출
=> array([0, 4, 8])
np.diag(a, k=1)    # k = 시작위치
=> array([1, 5])

np.random.uniform(0, 1, 10) # 균등분포 0~1값 10개 랜덤 추출
np.random.normal(0, 1, 10)  # 정규분포에서 값 10개 랜덤 추출  
```
### Axis 매우 중요!!
- np 계산 시 axis는 뒤로 한 칸씩 민다고 보면 됨
- (3, 4) = axis기준 (0, 1)  = (axis=0) : 3 / (axis=1) : 4
- (2, 3, 4) = axis기준 (0, 1, 2) = (axis=0) : 2 / (axis=1) : 3 / (axis=2) : 4 
```python
a = np.arange(12).reshape(3,4)
=>
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

a.sum(axis = 1) # (3, 4)에서 4기준으로
=> array([0+1+2+3, 4+5+6+7, 8+9+10+11])
a.sum(axis = 0) # (3, 4)에서 3기준으로
=> array([0+4+8, 1+5+9, 2+6+10, 3+7+11])
```
### Concatenate
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.vstack((a,b))        # 위아래로 붙이기
=>
array([[1,2,3], [2,3,4]])

a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])
np.hstack((a,b))        # 옆으로 붙이기 
=> array([[1, 4], [2, 5], [3, 6]])

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.concatenate((a,b))
=>
array([1, 2, 3, 4, 5, 6])

# axis = 0
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])

np.concatenate((a,b), axis=0)
=>
array([[1, 2, 3],
       [4, 5, 6]])

# axis = 1
a = np.array([[1, 2],[3,4]])
b = np.array([[5, 6]])

np.concatenate((a, b.T), axis=1)
=>
array([[1, 2, 5],
       [3, 4, 6]])
# 사칙연산 : 같은 위치끼리 계산해줌(+, -, *, /)
np.array([1,2,3]) * np.array([1,2,3])  = np.array([1,4,9]) # a.dot(b) 행렬곱과 다름
```

# broadcasting(numpy) : 같은 위치의 값끼리 더해주기
```
# vector + scaler = matrix 각 값에 scaler를 계산해줌
np.array([1, 2, 3]) + 3 
=> np.array([4, 5, 6])

# matrix + scaler
np.array([[1, 2, 3,], [1, 2, 3]]) + 3
=> np.array([[4, 5, 6], [4, 5, 6]])

# matrix + vector
np.array([[1, 2, 3,], [1, 2, 3]]) + np.array([1, 2, 3])
=> np.array([[2, 4, 6,], [2, 4, 6]]) 

# 같은 답
np.array([[0],[10],[20],[30]]) + np.array([0,1,2])
np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20], [30, 30, 30]]) + np.array([0,1,2])
np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20], [30, 30, 30]]) + np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2])
=>
array([[ 0,  1,  2],
       [10, 11, 12],
       [20, 21, 22],
       [30, 31, 32]])
```
![broadcasting](https://user-images.githubusercontent.com/110000734/191048528-ad8887f4-1c91-4fe3-8fc5-b327bd26d559.JPG)

### numpy 심화
```python
a = np.arange(4)
=> array([0, 1, 2, 3])

a>1
=> array([False, False, True, True])

np.any(a>1) # True (하나라도 만족하면 True)
np.all(a>1)  # False (하나라도 틀리면 False)
```
### np.where(조건, 참값, 거짓값)
```python
a = array([3, 3, 2])
np.where(a > 2, 3, 2)
=>
array([3, 3, 2])  # [(3>2, o) 3, (3>2, o) 3, (2>2, x) 2]

np.where(a>0) # 참,거짓 설정 안하면 인덱스 반환
=>
(array([0, 1, 2]),) # [(3>0, o) 0, (3>0, o) 1, (2>0, o) 2]
```
### argmax & argmin
```python
a = np.array([1, 2, 32, 4])
np.argmax(a), np.argmin(a)
=>
(2, 0) # (최고값 32인덱스, 최저값1의 인덱스)

a = np.array([[1, 2, 4, 7],
              [9, 88, 6, 45],
              [9, 7, 3, 4]])
np.argmax(a, axis=1), np.argmin(a, axis=0)
(array([3, 1, 0]), array([0, 0, 2, 2]))
``` 
### Bollean index
- binary형으로 만들 때 유용함!

```python
test = np.array([1, 4, 0, 2, 3, 8, 9 ,7], float)
test>3
=>
array([False,  True, False, False, False,  True,  True,  True])

test[test>3] # 조건에 맞는 인덱스의 값만 출력(where은 인덱스뽑기)
=>
array([4., 8., 9., 7.])

test_a = test>3 # 위에랑 같은거
test[test_a]
array([4., 8., 9., 7.])

# binary 만들기
a = np.array([1, 2, 3, 4])
b = a >2
b
=>
array([False, False,  True,  True])

b.astype(int)
=>
array([0, 0, 1, 1])
```
### fancy index & take
```python
# fancy
a = np.array([1, 2, 3, 4], float)
b = np.array([0, 0, 1, 2, 3], int)
a[b]
=>
array([1., 1., 2., 3., 4.]) # b의 값을 인덱스로 활용([a의0, a의0, a의1, a의2, a의3])

# take : 직관적으로 보이기 때문에 이 함수 쓰길 추천
a.take(b)
array([1., 1., 2., 3., 4.])

# Matrix
a = np.array([[1, 4],[9, 16]], float)
b = np.array([0, 0, 1, 1, 0], int)
c = np.array([0, 1, 1, 1, 1], int)
a[b,c]  # zip으로 묶어주는 개념 / array(a[0,0], a[0,1], a[1,1]...)    
=>
array([ 1.,  4., 16., 16.,  4.])    
```
### 함수
- df['columns'].shift(1) = 컬럼의 값을 인덱스 1칸씩 밀려서 받음
