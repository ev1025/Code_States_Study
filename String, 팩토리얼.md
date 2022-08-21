# String
- 영문 한 글자는 1byte의 메모리공간을 사용
- 1byte = 8bit = 2^8 = 256까지 저장가능
- 문자열에서 ‘나 “ 사용방법 :  ”It’s ok“ 나 ‘It\’s ok’
- 두 줄 이상 쓰는 방법 : \n나 글 앞 뒤에 ”“” 

```python
.capitalize()      # 첫문자를 대문자형으로 전환
.title()           # 띄어쓰기 후 한 글자만 대문자
.Upper()           # 대문자(isupper : 대문자인지 반환)
.lower()           # 소문자(islower : 소문자인지 반환)
.strip()           # 좌우 공백을 없앰
.split(‘’)         # ‘’을 기준으로 잘라서 List로 바꿔줌
.isdigit()         # 숫자가 있는지 확인
.startswith(‘abc’) # 숫자열 abc로 시작하는 문자 있는지 확인
.endswith(‘abc’)   # 문자열 abc로 끝나는 문자 있는지 확인
.count(‘abc’)      # abc 들어간 문자열 수 반환
.rfind(‘abc’)      # 문자열 abc 들어간 위치 반환
```

##### Call by value
- 함수에 인자를 넘길 때 값만 넘김, 인자값 변경 시 호출자에게 영향을 주지 않음    
- f(x) = ~~~ / a= 5 / f(a) = ~~ 일 때, x값이 7로 바뀌어도 a는 영향없음   
##### Call by reference 
- 함수에 인자를 넘길 때 메모리 주소를 넘김. 함수 내에 인자값 변경 시 호출자의 값도 변경 됨 (위에서 x값이 바뀌면 a값도 바뀜)
##### Call by Object Reference  
- 전달 된 객체를 참조하여 변경 시 호출자에게 영향을 주나, 새로운 객체를 만들 경우 호출자에게 영향을 주지 않음

----

## 변수의 범위(Scoping Rule)
- 지역변수(local variable) : 함수 내에서만 사용되는 변수(t)
- 전역변수(global variable) : 프로그램 전체에서 사용하는 변수(x) global x로 설정할 경우

```python
def test(t):
	print(x)
	t=20
	print(t)
x = 10
test(x)
print(t) # t가 선언이 안되어 있어서 미출력
10
20
에러  # 글로벌 함수 t가 등록되지 않았기 때문에 미출력
```
#### 글로벌함수
```python
def f():
	s = I love London
	print(s)

s = I love Paris
f()              # 지역변수 출력
print(s)         # 글로벌 변수 출력

<출력>
I love London
I love Paris
```

##### global s를 선언해주면
```python
def f():
	global s       # s = 함수 밖의 s를 나타냄
	s = I love London
	print(s)

s = I love Paris
f()              # 지역변수 출력
print(s)         # 지역변수 출력

<출력>
I love London
I love London
```
---

## recursive function(재귀함수)
- 재귀 종료조건까지 자기가 자신을 호출하는 함수   
- 팩토리얼 def와 for loop으로 만들어보기
#### def
```python
def factorial(x):
    if x==1:
        return 1
    else:
        return x*factorial(x-1)
x=5
factorial(x)
```
#### for loop
```python
a = int(input("팩토리얼을 구할 숫자를 입력하세요."))
x = 1
for i in range(1,a+1,1):
    x *= i
print (x)
```
---

#### doc string
- 파이썬 함수에 대한 상세 스팩을 사전에 작성(함수 사용자의 이행도up)
- 함수명 아래에 적어줌 \``` \``` 변수값, 리턴값 힌트 써주는 것   
#### 함수 작성 요령
- 함수는 가능한 한 짧게 작성
- 함수의 이름에 함수의 역할, 의도가 명확히 드러나게 작성
- 인자로 받은 값은 복사해서 사용[:]  list -> list_temp =  list[:] 
- 복잡한 수식은 함수로
- 파이썬 코딩 컨벤션(일관성 있는 코딩)
- 한 줄은 79자를 넘기지 말기
