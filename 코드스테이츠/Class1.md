# 클래스와 객체
### Object-Oriented Programming, OOP
- 객체 : 실생활에서 일종의 물건 속성(Attribute)와 행동(Action)을 가짐
- OPP는 이러한 객체 개념을 프로그램으로 표현 속성은 변수(Variable), 행동은 함수(method)로 표현 됨
- 클래스(class) : 붕어빵 틀 (설계도)
- 인스턴스(instance)(객체) : 붕어빵(실체)

### Class
- snake_case : 띄워쓰기 대신에 “_”를 사용하는 것(함수명, 변수명에 사용
- CamelCase : 띄워쓰기 부분에 대문자 (Class명에서 사용)
- Attribute 추가는 __init__, self와 함께(_init__은 객체 초기화 함수)
- __는 특수한 예약함수나 변수, 함수명을 변경(맨글링)하는데 사용
![캡처](https://user-images.githubusercontent.com/110000734/189044863-06889575-54da-4d20-a1e8-f9ec88fc0b3a.JPG)



```python  
class SoccerPlayer(object):   # 클래스예약어, 클래스이름, (상속받는 개체명) 상속은 안적어줘도 자동으로 생성됨
    def __init__(self, name, position, back_number): # 타입을 지정하고싶으면 (name : str , position : str, back_number : int):
    self.name = name
    self.position = position
    self.back_number = back_number

son = SoccerPlayer("흥민",  "fw", 7)    # 클래스에 인자를 넣어줌
park = SoccerPlayer("지성", "wf", 13)   

print(park)                             # 바로 print 하면 메모리 주소값만 출력됨
>> result 메모리주소
```
- 글자로 프린트하려면
```python
    def __str___(self):          
        return "hi My name is %s, My back number is %d"%\(self.name, self_back_number)
        
print(park)
>> result
Hi My name is 지성. My back number is 13
```
- Self란?
```python
    def change_back_number(self, new_number): # 함수 내에서는 본인을 지칭
        print("번호 변경 : from %d to %d" %(self.back_number, new_number))
        self.back_number = new_number
        
son.change_back_number(5)  # son = self 함수 밖에서는 객체(instance)이름을 지칭
>> result
번호 변경 : from 10 to 5
```
- add함수
```python
    def __add__(self, other):          # add는 클래스 값들을 합쳐줌 
        return self.name + other.name
        
son + park
>> result
'sonpark'
``` 
#### 클래스 내의 다양한 method
https://corikachu.github.io/articles/python/python-magic-method 


