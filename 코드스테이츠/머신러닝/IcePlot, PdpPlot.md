# Ice plot / Pdp plot

## Ice Plot
- Individual Conditional Expectation
- 특정 특성값 변화에 따른 모델의 예측량 변화
- 특정 행의 다른 특성은 고정, 하나의 특성만 min~ max로 변화를 주어 예측값 비교
- 각 행별 특성에 따른 변화(세부적인 변화 확인할 때 사용)
#### 특정 행의 특성(feat)의 min~max 변화에 따른 예측치 변화량
```python
target_point = x_test.iloc[[100]]  # 특정 행을 지정(나머지 특성 모두 고정)
                                   # 지정한 행의 특성(feat) 1개의 최소 ~ 최대값 범위
target_range = range(x_test['feat'].min(), x_test['feat'].max()+1)

results = [] # 변화량을 넣어줄 변수

for a in target_range:
    target_point['feat'] = a                                # 특정 행의 특성 값을 최소~ 최대값으로 넣음          
    x_test_pred_proba = rf.predict_proba(target_point)[:,1] # 특정 행의 특성 값이 변할 때 예측치
    results.append(x_test_pred_proba)

results = np.array(results)  
results -= results[0]        # results - results[0] 예측치의 변화량(각 값 - min값)
```
#### Ice Plot 시각화
```python
plt.plot(target_range, results)  # x축 : 특성의 범위, y 축 : 예측 변화량
plt.xlabel("feat")               # 특성
plt.ylabel("pred_proba")         # 예측변화량
```
![iceplot](https://user-images.githubusercontent.com/110000734/191513419-ed2ae7aa-e278-4863-b246-4c73f5915200.png)

## Pdp Plot
- Iceplot의 개별 값들의 평균(특성의 전반적인 이해와 판단)
- 세세한 조건의 예측변화를 확인하기 어려우나 전반적인 모델의 특성에 대한 이해를 할 수 있음
- 특성간의 독립성을 전제로 사용한다.
- 특정 특성의 복잡한 양상을 이해하기 쉽게 시각화해준다.
- 대신 다중공산성(특성간의 상관관계)을 파악하기 힘들다. (상관관계가 높을 경우 잘못된 결과가 나오기때문에 조심해야한다.)
- 데이터가 부족한 부분(분포)의 예측값은 부정확한 값이 생성될 가능성이 높으므로 데이터 분포를 보면서 분석해야한다.
### pdp plot(1-feature)
- 모델생성
```python
!pip3 install pdpbox
```
```python
plt.rcParams['figure.dpi'] = 144
from pdpbox.pdp import pdp_isolate, pdp_plot # pdpbox 하나의 특성에 대한 pdp시각화

target_feature = "JobSatisfaction"           # 특성 선택

isolated = pdp_isolate(              # pdp모델
    model = rf,                     # 모델
    dataset = x_test,               # x데이터
    model_features= x_test.columns, # 특성 목록
    feature = target_feature,       # 분석할 특성
    grid_type = "percentile",       # 특성 분포로 점 간격 설정(equal 개수만큼 동일하게 구분)
    num_grid_points = 10,           # 점의 개수(type에 따라 달라짐)
    # cust_grid_points=[-2, 1, 2, 3, 4, 5, 6, 7],  # 특성값을 찍어 볼 지점을 직접 지정
)
```
- pdp plot(파란색: 신뢰구간)
```python
pdp_plot(isolated,                      # pdp plot (파란색은 신뢰구간)
         feature_name = target_feature) # 그래프 위에축 이름
```
![2](https://user-images.githubusercontent.com/110000734/191513502-a041be5a-bbde-4175-876d-15fb91b2c3dd.png)

- pdp plot + ice plot
```python
pdp_plot(                 
    isolated,                      # isolate 함수(pdp함수)
    feature_name = target_feature, # 그래프 위,아래 표시할 feature name
    plot_lines = True,    # 개별라인 표시 여부(노란색)
    frac_to_plot = 50,    # 표시할 ice plot 수
    plot_pts_dist=True    # rug plot(데이터의 밑에 수량바) 데이터 분포 표시
)
```
![3](https://user-images.githubusercontent.com/110000734/191513650-7422a949-2ff2-4eec-b709-625f37f2cb2b.png)

### pdp plot(2-features)
- pdp모델 생성
```python
from pdpbox.pdp import pdp_interact, pdp_interact_plot # 2- feature pdp

target_features = ['JobInvolvement','TotalWorkingYears']

ineraction =pdp_interact(        # 2개의 특성 pdp
    model = rf,                  # 모델
    dataset = x_test,            # 사용데이터
    model_features = x_test.columns, # 특성들 이름(반드시 list나 tuple?)
    features = target_features,      # 사용할 
)
```
- pdp모델 시각화(2-features)
```python
pdp_interact_plot(ineraction,           # 히트맵으로 나타남
                  plot_type=  'contour',# 그래프 방식 'grid'는 히트맵
                  feature_names = target_features)
```
![222](https://user-images.githubusercontent.com/110000734/191513692-c333e4a9-34b9-43bb-bfcf-cc52487df459.png)

### mapping(범주형 데이터 pdp)
```python
# 인코딩 된 모든 값들 보기 (rf[0]가 오디널인코딩)
mappings = rf[0].mapping  
print(mappings)
```
```python
# 특정 특성 인코딩 값 보기(컬럼명, 타입)
mapping_data = list(filter(lambda x: x["col"] == 'Gender', mappings)) 
print(mapping_data)
```
- 인덱스 : 값 시리즈 보기
```python
maps = mapping_data[0]["mapping"] # mapping 주소에 저장
print(type(maps))
```
- 인코딩 전, 후 값 확인
```python
original_value = maps.index.tolist()  # 범주형 인코딩 전 값
encoding_value = maps.values.tolist() # 범주형 인코딩 후 값

print(encoding_value, original_value)
```
- 범주형pdp 모델생성 및 x축 범주로 라벨지정
```python
pdp_dist = pdp_isolate(
    rf,                                     # 모델
    x_test,                                 # 데이터
    model_features=x_test.columns,
    feature = 'Gender',                     # 범주형 특성
    cust_grid_points = [-2, 1, 2],          # x값 지정
)
pdp_plot(pdp_dist, 'Gender')                # 모델, 범주형특성
plt.xticks(encoding_value, original_value)  # x축에 특성 표시
```
![다운로드](https://user-images.githubusercontent.com/110000734/191513866-a640cd00-8a12-4376-8b4f-7713e0df7e3b.png)
