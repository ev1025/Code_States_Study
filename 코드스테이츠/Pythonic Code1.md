# Pythonic Code
- 타인의 코드를 이해하기위해서 코드쓰는 방법을 알아야함
- split : 주로 데이터를 자르는 용도
- join : 주로 데이터를 합치는 용도

### list comprehension
```python
result = []
for I in range(10):
    result.append(i)
>>>> result
[0,1,2,3,4,5,6,7,8,9]

result = [i(a) for i(b) in range(10)]          # 위와 같은 식 축소판 i(b)의 값을 i(a)에 넣어줌
>>>> result  
[0,1,2,3,4,5,6,7,8,9]

result = [i for i in range(10) if  i % 2 ==0]  # 뒤에 if조건식 작성가능 (if문을 필터라고 부름)
>>>> result = [0,2,4,6,8]
```

### Nested For loop
- 2개의 for loop 동시에 사용
```python
word_1 = ‘Hello’
word_2 = ‘World’
result = [ i + j for i in word_1 for j in word_2 ] # for loop 1번실행 후 2번실행
>>> result                                        #(1+1, 1+2, 1+3), (2+1, 2+2, 2+3)... 
[‘HW’, ‘Ho’, ‘Hr’, ‘Hl’, ‘Hd’, ‘eW’, ‘eo’, ‘er’.... ‘ol’,,‘od’ ]
```

### Filter
- Nested에 조건 걸어주기
```python
C_1 = [ ‘A’, ‘B’, ‘C’]
C_2 = [ ‘D’, ‘E’, ‘A’]
result = [ i + j for in c_1 for j in c_2 if not (i==j) ] # i == j가 아닌 데이터만 추가 
>>>>result
[‘AD’, ‘AE’, ‘BD’, ‘BE’, ‘BA’, ‘CD’, ‘CE’, ‘CA’ ]

[ I + j if not(i==j) else i for i in C_1 for j in C_2]   # not(i==j)면 i + j 아니면 i 반환 
```
### Two dimentional list(2차원 목록)
```python
words = "The quuck brown fox jumps over the lazy dog".split() # 단어별 구분
pprint.pprint([ [w.upper(), w.lower(), len(w)] for w in words ]) # 2차원으로 나눠서 저장
>>>result
[[ 'THE', 'the' 3],
 [ 'QUICK', 'quick', 5]... 

-----------------------------

C_1 = [ ‘A’, ‘B’, ‘C’]
C_2 = [ ‘D’, ‘E’, ‘A’]
[[i + j for i in  C_1] for i in C_2]                      # C_2가 먼저 실행, 2차원으로 생성
>>>>result
[ ['AD', 'BD', 'CD'], ['AE', 'BE', 'CE'], ['AA', 'BA', 'CA'] ] # C_2 고정

-------------------------------

[ [i + j for i in C_1 if  i != 'C' ] for i in C_2]            # 2차원 if문 작성요령
>>> result
[['AD', 'BD'], ['AE', 'BE'], ['AA', 'CA']]    
```


### enumerate : list의 element를 추출할 때 번호를 붙여서 추출 (index element)
```python
for i, v enmerate("ABC"):       # index, element 형식으로 추출
    print(i, v)
>>> result
0 A
1 B
2 C
------------------------------------
my_str = "ABCD"
{v : i for i, v in enumerate(my_str)}
>>> result
{'A' ; 0, 'B' : 1, 'C' : 2, 'D' : 3}
---------------------------------
list(set(text.split())    # text 단어를 추출해서 / set으로 중복 제거 후 / list생성
```
### zip
- 두 개의 list의 값을 병렬적으로 추출함
```python
alist = ["a1", "a2", "a3"]
blist = ["b1", "b2", "b3"]

[ [a,b] for a, b in zip(alist, blist) ]
>>> result
[['a1', 'b1'], ['a2', 'b2'], ['a3', 'b3']]           # List로 묶어줌

for i, values in enumerate(zip(alist, blist)):   # 인덱스, 그룹
    print(i, values)
[(0, ('a1','b1')), (1, ('a2','b2')), (2, ('a3', 'b3'))]
-----------------------------------
[ c for c in zip(alist, blist) ]                   # Tuple로 묶어줌
>>> result 
[('a1', 'b1'), ('a2', 'b2'), ('a3', 'b3')]   
---------------------------------------------------
math = (100, 90, 80)
kor = (90, 90, 70) 
eng = (90, 80, 70)

[sum(value) / 3 for value in zip(math, kor, eng)]
>>> result
[93.33333333, 86.6666666666, 73.3333333333]

```
----
### Lambda
- 함수 이름 없이, 함수처럼 쓸 수 있는 익명함수
- 수학의 람다 대수에서 유래함
- 코드해석이 어려워서 사용을 지양함
```python
f = (lambda x,y : x + y)                 # lamda 인수 : return
(lambda x,y : x + y)(10, 50)              # 60 

(lambda x : "-".join(x.split()))("My Happy") # ‘My-Happy'
```
### map
- 두 개 이상의 list에도 적용 가능
- 실행시점에 값을 생성하여 메모리 효율적(lambda와 마찬가지로 지양)
```python
ex = [1,2,3,4,5]
f = lambda  x : x**2

list(map(f, ex))           # 반드시 list사용해야함
>>> return
[1, 4, 9, 16, 25]          # [f(value) for value in ex]도 같은결과

list(map(f(x) if x %2 == 0 else x, ex))  # map으로 필터 사용
[ x**2 if x % 2 ==0 else x for x in ex] # for loop 으로 필터 사용
>>> return
[1, 4, 3, 16, 5]

g = lambda x, y : x + y
list(map(g, ex, ex))
>>> return
[2, 4, 6, 8, 10] 
```

### reduce
- list에 똑같은 함수를 적용해서 통합
```python
from functools import reduce
reduce(lambda x, y : x + y, [1, 2, 3, 4, 5]) # 앞의 두 개를 더해서 x를 최신화
>>> result
15  ( 1+2 -> 3+3 -> 6+4 -> 10+5)
```
-----------------
### iterable object
- 내부적 구현으로 _iter_ 와 _next_가 사용됨
- iter(), next()함수로 iterable 객체를 iterator object로 사용
```python
cities = [ 'seoul', 'bussan', 'jeju']
memory_address_cities = iter(cities)  # 메모리값을 memory_address_cities에 저장
next(memory_address_cities)         # iter() 는 메모리값을 주소에 저장
'seoul'
next(memory_address_cities)         # nex()는 저장된 메모리에서 순서대로 꺼내서 출력
'bussan'
next(memory_address_cities)
'jeju'
```
