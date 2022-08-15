

### if함수
```python
if<조건>:
    조건문
else:
    수행문
 
 x is y  =  x == y    # is는 메모리주소를 비교하는 것 -5~ 256에서는 같은 값이 나온다.
 x is not y = x != y  # 그 외의 메모리주소에선 같은 값이라도 False나올 수 있음
 if 1                 # 0은 존재하지 않는다, 1는 존재 한다는 뜻
 
 all(리스트)           # and 조건문 리스트가 모두 True면 true 아니면 False
 any리스트)            # any 조건문 리스트가 하나라도 True면 True
 
 valuse =12   # 삼항 연산자(Ternarny operator)
 is_even = True if value % 2 == 0 else False 
 
 
 if score >= 90: grade = 'A'
 if score >= 80; grade = 'B'  #  일 때, score가 90점 이어도, B로 출력됨
 =>
 if score >= 90; grade = 'A'
     elif score >=80; grade = 'B'
 else : grade = 'F'
```
```python
year = int(input("당신이 태어난 년도를 입력하세요.")) # input 함수는 입렵값 str로 반환
date = 2021 - year +1
   
if date>= 17 and date < 20:   # all(17<= date, date<20)도 가능
        mes = "고등학생"
elif date>= 14 and date <17:  #  (14 <= date and date < 17)도 가능
        mes = "중학생"
elif date>= 8 and date <14:
        mes = "초등학생"
else: mes = "학생이 아닙니다."
print(mes)

```

### for loop함수   
- 주로 i, j, k 값을 많이 사용한다.
- 반복문에 사용한다.
- 주로 0부터 시작한다.
- 정확한 반복 횟수를 알 때 사용(모를 때는 while 사용)
```python
for 변수명 in [1,2,3,4,5]:
  print(변수명)            # hello hello hello hello hello

for 변수 in range(0,5):
  print(변수명)            # 위와 같은 결과
  
range(1,4)                 # [1,2,3] 생성(뒤에 숫자 앞까지만 생성)
list(range(5))             # [0,1,2,3,4]
```
```python
for di in range(2):
    print (f"{di} : 바보")  # 0 : 바보 ; 1 : 바보

for i in "바보들":          
     print(i)               # 바 ; 보; 들
     
for i in ['바보들','멍청이','똥개']: 
    print(i)                # 바보들 ; 멍청이; 똥개
    
for i in range(1,10,2):
    print(i)                # 1; 3; 5; 7; 9

for i in range(10,1,-1):
    print(i)                 # 10; 9; 8; 7; 6; 5; 4; 3; 2 마지막숫자 1은 안나옴
```
### while문   
- 조건이 만족하는 동안 반복 명령문을 수행한다.
- 정확한 반복 횟수를 모를 때 사용한다.
```python
i = 0
while i <10:                  # i 가 10이 되면 멈춤
  print(f"{i} : Hello")
  i += 1                      # 1씩 증가시켜라, i = i + 1도 같음
  => Hello 인덱스 0~9까지 출력
```
### 반복문 제어    
#### break문 
```python
for i in range(4):  
   if i==2:
        break                  # break : 중지
    print(i)
print("end")                   # 0; 1; 2; end
```
#### continue문
```python

for i in range(5):
  if i==3:
      continue                  # countinue : 스킵하고 진행
  print(i)
print("end")                    # 0; 1; 2; 4; end (i==3은 스킵됨)
```

### 구구단 만들기   

```python
j = int(input("구구단 몇단을 계산할까요?"))
for i in range(1,10):
    gugu = j * i
    print(f"{j} X {i} = {gugu}")    # gugu 지우고 j*i도 가능
```
```python
i= 111
while i !=0:
    i = int(input("구구단 몇단을 계산할까요?"))    
    if 10>i and i>0:
        for j in range(1,10):
            print(f"{i}X{j} = {i*j}")
    elif (9<i or i<1) and i!=0:
        print("잘못입력하셨습니다.")
    else:
        print("게임이 종료되었습니다.")

 ```

### 이진수 만들기  
```python
sen = "12345"
aa = ""
for char in sen:
    aa = char + aa
    print(aa)              # 1; 21; 321; 4321; 54321
    
이진수 만들기   
i = int(input("십진수를 입력하시오"))
result =''
while i>0:
        나머지 = i%2
        몫 = i//2
        i =i//2
        result = str(나머지)+result
print(f"이진수 :{result}")
```
### 숫자맞추기 게임
```python
import random
true_num = random.randint(1,100) # 1~100의 임의의 숫자 생성

print("숫자를 맞춰보세요(1~100)")
i=1111                           # 임의로 값 지정(의미는 없음)
while true_num != i:
    i = int(input())
    if true_num > i:
        print("숫자가 너무 작습니다.")
    elif true_num<i:
        print("숫자가 너무 큽니다.")
    else:
        break
print(f"축하합니다!! {true_num} 정답입니다.")
```

