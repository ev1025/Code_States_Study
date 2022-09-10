## Decorator(@)
#### first-class objects
- 일등함수, 일급 객체
- 변수나 데이터 구조에 할당이 가능한 객체
- 파라메터로 전달이 가능 + 리턴 값으로 사용
- 파이썬의 함수는 일급함수
```python
def abc(x):
    return x*x
f = abc
f(5) = 25
```
- 함수를 파라미터로 사용(method에 square나 cube 넣으면 실행)
```python
def square(x):
    return x*x
def cube(x):
    return x*x*x
def formula(method, argument_list):
    return [method(value) for value in argument_list]
``` 
#### Inner function
- 함수 내에 또 다른 함수가 존재
- closures : inner function을 return값으로 반환
```python
def print_msg(msg):
    def printer():
        print(msg)
    return printer
another = print_msg("Hello, Python")
another()
```
#### decorator function
- 복잡한 closures 함수를 간단하게 만들어 줌
```python
def star(func):
    def inner(*args, **kwargs):
        print("*" * 30)
        func(*args, **kwargs) 
        print("*" * 30)
    return inner
    
@star            # star함수의 func파라미터로 들어감
def printer(msg):
    print(msg)
printer("hello")
>> result
******************************
hello
******************************


def star(func):
    def inner(*args, **kwargs):
        print(*args[1] * 30)
        func(*args, **kwargs) 
        print(*args[1] *  30)
    return inner
    
@star
def printer(msg,mark):
    print(msg)
printer("hello", "T")
>> result
T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T
hello
T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T


def star(func):
    def inner(*args, **kwargs):
        print("*" * 30)
        func(*args, **kwargs) 
        print("*" * 30)
    return inner
    
def bucks(func):
    def inner(*args, **kwargs):
        print("@" * 30)
        func(*args, **kwargs) 
        print("@" * 30)
    return inner  
    
@star
@bucks
def printer(msg):
    print(msg)
printer("abc")
>> result
******************************
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
abc
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
******************************
```
### decorator 응용
```python
def generate_power(exponent):       #1 exponent에 2 대입
    def wrapper(f):                 #2 f에 raise_two(7) 대입
        def inner(*args):      
            result = f(*args)       #3 f(*args) = raise_two(7)
            return exponent**result #4 2**(7**2) 
        return inner
    return wrapper
@generate_power(2)    # 인자값 지정해주고
def raise_two(n):     # 함수
    return n**2
print(raise_two(7)) 
>> result
562949953421312
```
## Polymorphism(다형성)
- 같은 이름의 메소드의 내부 로직을 다르게 작성
- draw(rectangle) = 삼각형 생김 or draw(Circle) = 원생김
```python
class Animal:
    def __init__(self, name):
        self.name = name
    def talk(self):
        raise NotlmplementedError   # 에러를 직접 일으킴
```
```python
# 부모클래스 상속
class Cat(Animal):
    def talk(self):
        return 'Meow!'
class Dog(Animal):
    def talk(self):
        return 'Woof!'
```
```python
# 출력
animals = [Cat('Missy'), 
           Cat('Mr. Mistoffelees'),
           Dog('Lassie')]
for animal in animals:
    print(animal.name + ' : ' + animal.talk())
>> result
Missy : Meow!
Mr. Mistoffelees : Meow!
Lassie : Woof!
```
### Visibility
- 객체의 정보를 볼 수 있는 레벨을 조절
- 객체 안에 변수를 볼 수 없도록 조절
- Encapsulation라고도 함( 캡슐화, 정보은닉)
- Class를 설계할 때 클래스 간 간섭 / 정보공유 최소화
```python
```python
class Product(object):
    pass
```
```python
class Inventory(object):
    def __init__(self):
        self.__items = []                 # __ = Private 변수로 선언하여 외부(타객체) 접근하지 못하게 함
        
    def add_new_item(self, product):
        if type(product) == Product:      # product 타입인지 확인
            self.__items.append(product)  # 값을 추가
            print("new item added")
        else:
            raise ValueError("Invalid Item")
            
    def get_number_of_items(self):
        return len(self.__items)
        
    @property              # 함수명을 변수명처럼 사용하여 Private변수를 출력 할 수 있게함
    def items(self):
        return self.__items
        
my_inventory = Inventory()

my_inventory.__items       # 객체 미출력
my_inventory.items         # 객체 출력(property때문에)
```
### 함수
```python
def - raise ValueError         # 오류를 직접 일으키는 함수 뒤에 에러이름 써주면 됨

df.set_index('A')              # A열을 인덱스로 사용
df.ix[7]                       # iloc은 0부터시작, ix 는 진짜 그 번호 출력(인덱스에 문자가 포함되면 iloc과 같아짐)

df.loc[df['그룹'].isna()]      # 그룹컬럼이 결측치인 행 출력
df.loc[df['그룹'].isna(), ['이름', '소속사']] # 그룹이 결측치인 행의 이름,소속사 컬럼 출력

df.insert(n, 'columns', 값)    # n+1번째에 columns 이름으로 값을 가지는 열 산입

df['columns'].str[:5]          # 앞에서 5글자 출력 (서울특별시)
df['columns'].str[5::-1]       # 거꾸로 출력 (시별특울서)

df.query("index > 1")          # 인덱스로 나누기
df.query('name.str.contain("a")')             # name에 a 포함하는 데이터
df.query('name.str.contain("a", case=False)') # case =False 대소문자 구분안함

df['columns'].str.split(" ", expand=True) # 분리된 List를 DF형식으로 출력해줌

np.linspace(2.0, 3.0, num=5, # 시작, 끝, 5등분
            endpoint=False,  # True는 끝 값 포함, False는 끝값 미포함
            retstep=True)    # 간격 출력(5개의 간격이 얼마인지)
```
