# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:49:36 2020

@author: 韩琳琳
"""

import pandas as pd

w=pd.read_csv('./Data/d1.csv')
w.columns=['id','href','blog','pubtime','like','comment','transfer']
w = w[w['blog'].notnull()]
isNA=w.isnull()
w=w.dropna()
w[isNA.any(axis=1)]
w = w.drop_duplicates()
w.sort_values(["pubtime"],ascending=True,inplace=True)
w['pubtime'] = pd.to_datetime(w['pubtime'])
w.pubtime.dt.strftime("%Y-%m-%d")
w['date'] = w['pubtime'].dt.date
w['date']=pd.to_datetime(w['date'])
h=w[w['date']<pd.to_datetime('20200203')]
t=h[h['date']>pd.to_datetime('20200126')]
t.to_csv('d2.csv',index=False,encoding='utf_8_sig')
#非洲猪瘟
y1 = w[w['blog'].str.contains('猪瘟')]
cutlist1 = list(y1.index)
x1 = y1[y1['blog'].str.contains('武汉')]
exc1 = list(x1.index)
cutlist1 = list(set(cutlist1)^set(exc1)) #列表求差集

cutlist2=[]
nos = ['甲肝','大肠杆菌','鼠疫','动植物','林业','口蹄疫']
for i in nos:
    y2 = w[w['blog'].str.contains(i)]
    #print(y2.blog)
    cut_ = list(y2.index)
    cutlist2.extend(cut_)
cutlist2 = list(set(cutlist2))
#饭圈撕逼
fans = ['朱一龙','肖战','一博','陈飞宇','迪丽热','罗云熙','饭圈','博君一肖','新剧'
        '星座','运势','占卜','反黑','王俊凯','甄嬛传','粉丝','投资','股市','大跌'
        ,'板块','鼠疫','波兰','愿望','地震','甲肝','利好','塔罗','转发微博']
cutlist3 = []
for i in fans:
    y3 = w[w['blog'].str.contains(i)]
    #print(y3.blog)
    cut_ = list(y3.index)
    cutlist3.extend(cut_)
cutlist3 = list(set(cutlist3))

y4 = w[w['id'].str.contains('旅行健康')]
cutlist4 = list(y4.index)

cutlist = set(cutlist1+cutlist2+cutlist3+cutlist4)
res = list(set(w.index)^cutlist)
result = w[w.index.isin(res)]
result.to_csv('./Data/d1.csv',index=False,encoding='utf_8_sig')


