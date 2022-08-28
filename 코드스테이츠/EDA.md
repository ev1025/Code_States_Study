# 파이썬공부 1일차(2022.08.02)
세상에... 1일1커밋을 실현하고자 브런치에 가입했는데.. 작가가 아니면 공개가 안된단다... 비전공 32세의 우당탕탕 개발자되기..
첫날 포기할까 고민했지만 동기들을 보며 마음을 바로잡았다.   
너무많이 놀란 학원.. 학원이라는 표현 자체가 맞을까 모르겠으나 일단은 다녀보기로...   
   
      
      

### EDA(Exploratory data analysis) 탐색적분석법
- 데이터를 다각도로 이해하는 과정!  

- feature engineering : 기존 feature들을 재조합하여 새로운 feature를 만드는 과정 ex) 중량, 가격을 알때 중량 당 가격을 도출

- pandas 데이터프레임에서 데이터를 분석하고 의미도출     

- numpy 수학적 계산을 위해 사용     

- matplotlib.pyplot 시각화       

- 매서드 : 함수보다 작은단위     

- 라이브러리 : 임의로 설정된 값    

- raw data  : 원본데이터   

- name error  : import 선언 안했을 때 나오는 에러     

- SyntaxError : 문법적오류(', ; 같은 것들)     

- ValueError  : 부적절한 값을 인자로 받았을 경우 (잘못된 값(value)를 가져왔을때)   

- Import error : import에서 임포트하려는 이름을 찾을 수 없을 때 

- 피쳐 =columns ,index는 0부터 시작한다    

### 함수들
```.duplicated()              # 중복값 확인(bool 형태 True or False로 표시됨)

.duplicated().sum()      # 중복값 갯수 확인

.drop_duplicates()       # 중복값 삭제 

.rest_index(drop=True)   # 인덱스 재정렬 False로 할경우 기존 열을 그대로 남겨둠, 
                           데이터를 csv로 변환할 때에는 인덱스 지정이 되지 않도록 index= false 지정
                           
.np.repeat('단어',len())  # 단어를 len()만큼 반복 , np.array(['red']*len(df_red))

df_white.append(df_red, ignore_index=True) # 레드와 화이트 합침 , df_white.concat(df_red)도 가능, = merge, join

df = pd.read_excel('url', 'sheet_name') # url주소 엑셀파일의 시트를 불러옴

df = df.T                  # 행과 열을 바꿈

df_columns = df1.loc[0]    # df_columns에 df1의 0행값을 지정

df.columns = df_columns    # df의 columns을 df_columns으로 지정

df = df[1:]                # df를 1행부터 표시(0행 제외)

df.head()                  # 첫 5행을 데이터프레임 형태로 보여줌)안에 숫자입력시 그만큼 보여줌
                              (-1)은 뒤에서 첫 번째 행(-는 0이 없음)
                                df.tail() 뒤에서5개 
                                
.shape()                   # (row.columns)     

.info()                    # 인덱스(row)갯수, 열(column)갯수,열제목 데이터 결측치와 null값은 결측치, 열의 타입

.isnull()                  # 결측치 (true false 보여줌)

.isnull().sum()            # 열별 결측치 개수 보여줌

리스트함수

LI - ['A','B','C','D']   

LI[0] = A 

LI[-1] = D  (뒤에서 1번째)     

loc / iloc 함수

[ 숫자 ] = 행(ROWS)가져옴

[ : , 숫자] = 열(COLUMNS) 가져옴



df.LOC[ :, 이름] 이름으로 찾음

df.ILOC[ : , 숫자] 숫자로 찾음          

파일=파일[:1] 1행부터 표시하기(0행 제외)

df = df[:1]          



내 컴퓨터의 파일 업로드

from google.colab import files

uploaded = files.upload()          


파일업로드
df = pd.read_excel('url', 'sheet_name')
