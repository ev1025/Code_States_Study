# Class 2
### 1. Inheritance(상속)
```python
class Person(object):  # (object) 안쓰면 디폴트 object
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return “ 내 이름은 {0}, 나이는 {1}”.format(self.name, self.age) 

class Korean(Person): # 클래스명(Person)을 상속받음
    pass

first_korean = Korean("jinwoo“, 32) # 그대로 클래스 사용
print(first_korean)
>>result
내 이름은 jinwoo, 나이는 32

calss Employee(Person):
    def __init__(self, name, age, gender, salary, hire_date): # 부모클래스 Person에서 상속
        super().__init__(name, age, gender) # super는 부모클래스의 객체 불러옴
        self.salary = salary       # 불러온 객체에 값 추가
        self.hire_data = hire_date  
    
    def about_me(self):
        super().about_me()
        print("급여는”, self.salary, “원이구요. 입사일은”, self.hire_date, "입니다.“)
        
# 속성 없을 때
myEmployee = Employee("진우”, 32, “Male")
myEmployee.about_me()  # 자식클래스 값이 없어서 안나옴?
>>result
내 이름은 진우, 나이는 32

# 속성 생겼을 때
myEmployee = Employee("진우”, 32, “Male", 30000, ”2000/04/01”) # 부모클래스가져옴
myEmployee.about_me()                                          # 자식클래스 가져옴
>> result
내 이름은 진우, 나이는 32                         # 부모클레스의 형식을 가져옴
급여는 30000원이구요. 입사일은 2000/04/01입니다. # 자식클레스의 형식을 가져옴
