import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
import seaborn as sns
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
import pickle
import joblib
import re

import warnings
warnings.filterwarnings('ignore')



class Vegan_Cosmetics:
    def __init__(self):
        self.gender, self.age = input('성별과 연령대를 입력하세요 : ').split()
        self.beauty = pd.read_csv('./data/total_beauty.csv')
        self.봄 = pd.read_csv('./data/total_봄.csv')
        self.봄['계절'] = '봄'
        self.여름 = pd.read_csv('./data/total_여름.csv')
        self.여름['계절'] = '여름'
        self.가을 = pd.read_csv('./data/total_가을.csv')
        self.가을['계절'] = '가을'
        self.겨울 = pd.read_csv('./data/total_겨울.csv') 
        self.겨울['계절'] = '겨울' 

    # 오늘의 날씨 데이터 수집
    def weather_i(self):
        req = requests.get('https://www.weatheri.co.kr/forecast/forecast10.php')
        # 한글이 깨지는 문제를 해결하고자 decode 사용
        soup = BeautifulSoup(req.content.decode('utf-8','replace'), 'html.parser')

        # 월, 일, 요일
        date = soup.find('font', color='#124d79').text
        year, month, day = date[:4], date[6:8], date[10:12]
        ymd = year + '-' + month + '-' + day
        ymd = pd.to_datetime(ymd)
        week = ymd.weekday()
        
        # 공휴일
        holiday = pd.read_excel('./data/국가공휴일2.xlsx')
        holi_idx = holiday[(holiday['년'] == int(year)) & (holiday['월'] == int(month)) & (holiday['일'] == int(day))].index
        if len(holi_idx) == 0:
            holi = 0   # 공휴일이면 1, 아니면 0
        else:
            holi = 1

        data = []
        # 강수량, 풍속, 체감온도 (서울)
        table = pd.read_html(str(soup.select('table')[10]))[0]
        table = table.rename(columns=table.iloc[0]).drop(table.index[0])
        table.reset_index(drop=True, inplace=True)
        table['풍 속(m/s)'] = table['풍 속(m/s)'].apply(lambda x: float(x)*3.6)
        table = table.rename(columns={'풍 속(m/s)':'풍 속(km/h)'})

        try: 
            re.search('^[+-]?\d*(\.?\d*)$', table.loc[0]['강수량(mm)'])
            rain = 0
        except:
            rain = table.loc[0]['강수량(mm)']
        wind = table.loc[0]['풍 속(km/h)']
        temp = float(table.loc[0]['현재기온(℃)'])
        a_temp = 13.12 + 0.6215*temp - 11.37*(wind**0.16) + 0.3965*(wind**0.16)*temp

        # 미세먼지
        req2 = requests.get('https://www.weatheri.co.kr/special/special05_1.php?a=1')
        soup2 = BeautifulSoup(req2.content.decode('utf-8','replace'), 'html.parser')

        table2 = pd.read_html(str(soup2.select('table')[11]))[1]
        table2 = table2.rename(columns=table2.iloc[0]).drop(table2.index[0])
        table2.reset_index(drop=True, inplace=True)

        pm10 = table2.iloc[0,2]

        data.append([month, day, week, holi, self.gender, self.age, rain, wind, a_temp, pm10])
        df = np.transpose(pd.DataFrame(data[0]))
        df.columns = ['월','일','요일','공휴일','성별','연령대','평균일강수량(mm)','평균풍속(km/h)','체감온도(℃)','일 미세먼지 농도(㎍/㎥)']
        return df
    
    # 오늘의 날씨를 포함한 테스트셋 생성
    def weather_testset(self):
        df2 = self.weather_i()
        for i in range(self.beauty.소분류명.nunique()-1):
            df2 = df2.append(pd.Series(df2[0:1].values[0], index=df2.columns), ignore_index=True)
        df2['소분류명'] = self.beauty.소분류명.unique()
        df2 = df2[['월','일','요일','공휴일','성별','연령대','소분류명','평균일강수량(mm)','평균풍속(km/h)','체감온도(℃)','일 미세먼지 농도(㎍/㎥)']]
        return df2

    # 10만건당 건수 모델에 따른 테스트셋 전처리
    def preprocessing(self, data):
        # 범주형 변수 레이블 인코딩
        le = LabelEncoder()
        le = le.fit(data['성별'])

        le2 = LabelEncoder()
        le2 = le2.fit(data['소분류명'])

        # 연속형 변수 스케일링
        nu = data.drop(['계절','성별','소분류명','10만건당 건수'], axis=1)
        scaler = StandardScaler()
        scaler = scaler.fit(nu)
        return le, le2, scaler
        
    # 계절별로 테스트셋에 전처리 적용
    def testset_preprocessing(self):
        df2 = self.weather_testset()
        
        if int(df2['월'][0]) in [3,4,5]: # 봄
            le, le2, scaler = self.preprocessing(self.봄)
        elif int(df2['월'][0]) in [6,7,8]: # 여름
            le, le2, scaler = self.preprocessing(self.여름)
        elif int(df2['월'][0]) in [9,10,11]: # 가을
            le, le2, scaler = self.preprocessing(self.가을)
        else: #겨울
            le, le2, scaler = self.preprocessing(self.겨울)
            
        # 범주형 변수 레이블 인코딩
        df2['성별'] = le.transform(df2['성별'])
        df2['소분류명'] = le2.transform(df2['소분류명'])
        label_df2 = df2[['성별','소분류명']]
        
        # 연속형 변수 스케일링
        nu = df2.drop(['성별','소분류명'], axis=1)
        scaled = scaler.transform(nu)
        scaled_df2 = pd.DataFrame(scaled, columns=nu.columns)
        
        df3 = pd.concat([scaled_df2, label_df2], axis=1)
        return le2, df3
    
    # 계절별로 10만건당 건수 예측 (ExtraTreesRegressor)
    def extratrees_social(self):
        le2, df3 = self.testset_preprocessing()
        
        if int(df3['월'][0]) in [3,4,5]: # 봄
            et_model = joblib.load('./model/ExtraTreesRegressor(봄)')
            pred = et_model.predict(df3)
        elif int(df3['월'][0]) in [6,7,8]: # 여름
            et_model = joblib.load('./model/ExtraTreesRegressor(여름)')
            pred = et_model.predict(df3)
        elif int(df3['월'][0]) in [9,10,11]: # 가을
            et_model = joblib.load('./model/ExtraTreesRegressor(가을)')
            pred = et_model.predict(df3)
        else: #겨울
            et_model = joblib.load('./model/ExtraTreesRegressor(겨울)')
            pred = et_model.predict(df3)
            
        df3['10만건당 건수'] = np.expm1(pred)
        df3['소분류명'] = le2.inverse_transform(df3['소분류명'])
        return df3[['소분류명','10만건당 건수']]

    # 구매건수 모델에 따른 테스트셋 전처리
    def preprocessing2(self, data):
        # 범주형 변수 레이블 인코딩
        le = LabelEncoder()
        le = le.fit(data['계절'])

        le2 = LabelEncoder()
        le2 = le2.fit(data['성별'])

        le3 = LabelEncoder()
        le3 = le3.fit(data['소분류명'])
        return le, le2, le3

    # 계절별로 테스트셋(예측한 10만건당 건수 포함) 전처리 적용
    def total_testset_preprocessing(self):
        df2 = self.weather_testset()
        df3 = self.extratrees_social()
        df4 = pd.merge(df2, df3, on='소분류명') 

        if int(df4['월'][0]) in [3,4,5]: # 봄
            le, le2, le3 = self.preprocessing2(self.봄)
            df4['계절'] = '봄'
        elif int(df4['월'][0]) in [6,7,8]: # 여름
            le, le2, le3 = self.preprocessing2(self.여름)
            df4['계절'] = '여름'
        elif int(df4['월'][0]) in [9,10,11]: # 가을
            le, le2, le3 = self.preprocessing2(self.가을)
            df4['계절'] = '가을'
        else: #겨울
            le, le2, le3 = self.preprocessing2(self.겨울)
            df4['계절'] = '겨울'

        # 범주형 변수 레이블 인코딩
        df4['계절'] = le.transform(df4['계절'])
        df4['성별'] = le2.transform(df4['성별'])
        df4['소분류명'] = le3.transform(df4['소분류명'])
        return le3, df4

    # 계절별로 구매건수(예측한 10만건당 건수 포함) 예측 (RandomForest)
    def randomforest_buy(self):
        le3, df4 = self.total_testset_preprocessing()

        if df4['계절'][0] == '봄':
            rf_model = joblib.load('./model/RandomForest(봄).pkl')
            pred = rf_model.predict(df4)
        elif df4['계절'][0] == '여름':
            rf_model = joblib.load('./model/RandomForest(여름).pkl')
            pred = rf_model.predict(df4)
        elif df4['계절'][0] == '가을':
            rf_model = joblib.load('./model/RandomForest(가을).pkl')
            pred = rf_model.predict(df4)
        else: #겨울
            rf_model = joblib.load('./model/RandomForest(겨울).pkl')
            pred = rf_model.predict(df4)

        df4['구매건수'] = np.expm1(pred)
        df4['소분류명'] = le3.inverse_transform(df4['소분류명'])
        df4 = df4.loc[df4.구매건수.sort_values(ascending=False).index].reset_index(drop=True)
        return df4[['소분류명','구매건수']]