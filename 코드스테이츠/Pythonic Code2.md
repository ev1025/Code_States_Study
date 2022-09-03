## generator
- iterable object를 특수한 형태로 사용해주는 함수
- element가 사용되는 시점에 값을 메모리에 반환 (메모리 절약)
 : yield를 사용해 한 번에 하나의 element를 반환
### generator 활용 용도
- list타입의 데이터를 반환해주는 함수
 : 읽기 쉬운 장점, 중간에 loop이 중간될 수 있을 때 사용
- 큰 데이터를 처리할 때
 : 데이터가 커도 처리의 어려움이 없음
- 파일 데이터를 처리할 때
```python
def general_list(value):        # value개의 숫자 함수 생성
    result = []
    for i in range(value):
        result.append(i)
    return result
--------------------------
def generator_list(value):
    result = []
    for i in range(value):
        yield i
generator_list(50)               # generator형태로 저장(주소값만 가지고있음)
for a in generator_list(50):     # 값을 보려면 for loop을 써야함
    print(a)

```

### generator comprehension
- generator expression 으로도 불림
- list comprehension 은 [], 이건 () 사용
```python
gen_ex = (n*n for n in range(500)) # generator 타입으로 변수생성(주소만 가짐)
```
## passing arguments
> 함수에 입력되는 다양한 arguments 형태
1) keyword argument
- 함수에 입력되는 parameter의 변수명을 사용, arguments를 넘김
- print 할 때 값을 지정해주는것
```python
def name(my_name, your_name):
   print(f"hi {0}, bye{1}")

print(your_name = "바보“, my_name = "멍청이”)   # 키워드에 값을 지정해주면 순서 무시
>>> result
hi 멍청이, bye 바보     # print(f" hi {0}, bye{1}“)로 하면 ’hi 바보, bye 멍청이‘로 출력
```
 2) default arguments
- parameter의 기본값을 사용, 입력하지 않을 경우 기본값 출력
- 함수 생성 시 기본값을 지정하는 것
```python
def name(my_name ,your_name= ’똥개‘):
    print(f"hi {0}, bye {1}")

name(바보)
>>> result
hi 바보, bye 똥개         # name(바보, 멍청이) ==> hi 바보, bye 멍청이
```
 3) 가변인자(variable-length)
- 개수가 정해지지 않은 변수를 함수의 parameter로 사용하는 법
- keyword arguments와 함꼐 argument 추가 가능
- Asterisk(*)기호를 사용하여 함수의 parameter를 표시함
- 입력된 값은 tuple type으로 사용할 수 있음
- 가변인자는 오직 한 개만 맨 마지막 parameter 위치에 사용가능
- 기존 parameter 이후에 나오는 값을 tuple로 저장함
### *args
```python
def asterisk_test(a, b, *args):
    return a+b+sum(args)
print(asterisk_test(1, 2, 3, 4, 5))     # a = 1 /  b= 2 /  *args = (3, 4, 5) tuple형태
>>> return
15
```
### **kwargs
- dict타입으로 반환
```python
def kwargs_test_1(**kawargs):
    print(kwargs)

kwargs_test_1(firts = 1, second = 2, third = 3)
>>> result
{ 'firtst' : 1, 'second' : 2, 'third' : 3 }
```

### 가변인자 순서
```python
def 함수명(기본, 디폴트args=2, *arge(가변인자), **kwargs(키워드가변인자)): # 순서

함수명(10, 30, 3, 5, 6, 7, 8, first = 1, second = 2 , third = 3)
>>> 기본 = 10 / 디폴트args = 30 / *args = (3, 5, 6, 7, 8)  / **kwargs ={ 'first' : 1, 'second' : 2 , 'third' : 3)


함수명(one=10, two=300, first = 1, second = 2 , third = 3)  # 키워드args로 넣어버리면 기본 사용안됨, *args는 생략가능함
>>> 기본 = 10 / 디폴트args = 300 / *args = () / **kwargs ={ 'first' : 1, 'second' : 2 , 'third' : 3)
```

### asterist(*) - unpacking a container
- tuple, dict 등 자료형에 들어가있는 값을 unpacking
- 함수의 입력값, zip 등에 유용하게 사용가능
```python
def asterisk_test(a, *args) # *가 여러개의 가변인자를 받아줌
   print(a, *args)
   print(a, args)

asterisk_test(1, *(2,3,4,5)) # *가 unpacking
>>> result
1, 2, 3, 4, 5       # (a, *args)
1, (2, 3, 4, 5)      # (a, args)
 
asterisk_test(1, (2,3,4,5))  # (2,3,4,5)를 1개의 변수로 지정
>>> result
1, (2, 3, 4, 5)       # (a, *args)
1, ((2, 3, 4, 5),)     # (a, args) 튜플 값 1개가 들어감
-----------------------------------------------------------
a, b, c = ([1, 2], [3, 4], [5, 6])  # unpacking
print(a, b, c)
a = [1, 2] , b = [3, 4], c = [5, 6]

data = ([1, 2], [3, 4], [5, 6])     # 위와 같음
print(*data)
-------------------------------------
def asterisk_test(a, b, c, d)
    print(a, b, c, d)

data = {"b" : 1, "c" : 2, "d" : 3}
asterisk_test(10, **data)          # 언패킹

>>> result 
10, b=1, c=2, d=3
```
### zip 응용
```python
ex = ([1, 2], [3, 4], [5, 6]) # 값이 2차원 tuple 1개
for value in zip(*ex):        # *가 ()를 없애줌(unpacking)
    print(value)              # *가 없으면 ([1,2],) / ([3,4],) / ([5,6]) 출력
>>> result
(1, 3, 5)
(2, 4, 6)
```
