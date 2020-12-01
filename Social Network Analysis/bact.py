# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:47:53 2020

@author: 韩琳琳
"""

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
import jieba
import networkx as nx
import math  
import community
import matplotlib.pyplot as plt
import numpy as np  
import time
import jieba.posseg 
from datetime import datetime
import os


def read(csv_name): #逐日分析
    w=pd.read_csv(csv_name)#或 header=0,不能写True
    w['pubtime'] = pd.to_datetime(w['pubtime'])
    w.pubtime.dt.strftime("%Y-%m-%d")
    w['date'] = w['pubtime'].dt.date
    d_range = w['date'].unique()
    return w,d_range
 
def preprocess(w): #  输出 da：文本预处理后的列表
    jieba.load_userdict("./Data/user_dict.txt")
    stopword=[]
    f = open('./Data/stopword_bact.txt', encoding='utf-8')
    for line in f :
        line=line.strip('\n')
        stopword.append(line)           
    da=[]
    for i in range(w.shape[0]):
        da_1 = []
        b = str(w.iloc[i,2]) 
        #b = str(w.iloc[i,0])     
        b = jieba.posseg.cut(b)
        for x in b:
            if (x.flag.startswith('v') and len(x.word)>1) or (x.flag.startswith('n') and len(x.word)>1):
            #if x.flag.startswith('n') and len(x.word)>1:
                 da_1.append(x.word)  
            #去除停用词：
        da_2=[]
        for word in da_1:
            if word not in stopword:
                da_2.append(word)
        filestr = ' '.join(da_2)
        da.append(filestr)
        
    return da
    

def NTDE_4(da,t1,t2): # t1 关键词w的出现次数/总词数 t2 包含关键词w的微博数/微博总数
    cv = CountVectorizer()
    cv_fit=cv.fit_transform(da)  
    vec = cv_fit.toarray()    
    nodes = cv.get_feature_names()
    col = vec.sum(axis=0) #关键词w的出现次数
    feature_word=[] 
    t1_result = col/col.sum()
    exist = (vec > 0) * 1.0
    #factor = np.ones(vec.shape[1])
    #res = np.dot(exist, factor) #每行非零个数
    num = exist.sum(axis=0) #包含关键词w的微博数
    t2_result = num/len(vec)
    for i, j, node in zip(t1_result, t2_result, nodes):
        #print(i,j,node)
        if i > t1 and j > t2:
            feature_word.append(node)     
    feature_word = list(set(feature_word))
    return feature_word
    

def TFIDF(da,t):
    cv = CountVectorizer()
    cv_fit=cv.fit_transform(da)   
    vec = cv_fit.toarray()    
    nodes = cv.get_feature_names()
    col = vec.sum(axis=0)
    feature_word=[]
    for i in range(len(vec)):
        print(i)
        for j in range(len(vec[i])):    
             tf = vec[i][j]/sum(vec[i]) 
             idf = math.log(len(vec)/col[j] )
             if tf*idf > t:
                 feature_word.append(nodes[j])
    feature_word = list(set(feature_word))
    return feature_word

def mygraph(feature_word,da,date,t1,t2):
    length = len(feature_word)
    adjacent = [[0 for i in range(length)] for j in range(length)]
    for line in da:
        line = line.split()
        for i in range(length):
            if feature_word[i] in line:
                adjacent[i][i] += 1
                for j in range(i+1,length):
                    if feature_word[j] in line:
                        adjacent[i][j] += 1  
    E = []
    for i in range(length):
        for j in range(i+1,length):
            try:
                weight = pow(adjacent[i][j],2)/adjacent[i][i]/adjacent[j][j]
                E.append((feature_word[i], feature_word[j], weight)) 
            except:
                pass
    G=nx.Graph()
    G.add_nodes_from(feature_word)
    G.add_weighted_edges_from(E)
    com_dict = community.best_partition(G)
    coms = [[i for i in com_dict.keys() if com_dict[i] == com] for com in set(com_dict.values())]
    color = []
    # 黄色'#ffd900'绿色 '#aacf53'灰紫'#D291BC'灰色'#9ea1a3'淡紫'#C1BBDD'水蓝'#59b9c6'
    #天蓝'#0095d9'肉色'#FFDFD3'珊瑚'#ee836f'绿色'#028760'银色'#adadad'浅绿'#7ebeab'
    color_dict = {0: '#ffd900', 1: '#aacf53', 2: '#D291BC', 3: '#ee836f', 4: '#C1BBDD', 5: '#59b9c6', 6: '#FFDFD3',
                  7: '#adadad', 8: '#7ebeab', 9: '#FEC8D8', 10: '#deb068', 11: '#aacf53', 12: '#028760', 13: '#dd7a56',
                  14: '#93ca76', 15: '#c0a2c7', 16: '#f5e56b', 17: '#4c6cb3',18: '#ffd900', 19: '#aacf53', 20: '#D291BC', 
                  21: '#ee836f', 22: '#C1BBDD', 23: '#59b9c6', 24: '#FFDFD3',25: '#adadad', 26: '#7ebeab', 27: '#FEC8D8',
                  28: '#deb068', 29: '#aacf53', 30: '#028760', 31: '#dd7a56',32: '#93ca76',33: '#c0a2c7', 34: '#f5e56b',
                  35: '#4c6cb3'}
    G_graph = nx.Graph()
    par = []
    sub_g = []
    for j, each in enumerate(coms):
        G_graph.update(nx.subgraph(G, each))
        color.extend([color_dict[j]] * len(each))
        par.append(each)
        sub_graph = G.subgraph(each)
        sub_g.append(sub_graph)
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    pos = nx.spring_layout(G_graph, seed=4, k=0.4)
    nx.draw(G, pos, with_labels=False, node_size=1, width=0.1, alpha=0.2, edge_color='#dcdddd')  # 白鼠
    nx.draw(G_graph, pos, with_labels=True, edge_color='#778899', node_color=color, node_size=70, width=0.5, font_size=5,
            font_color='#000000')  
    plt.savefig(f'./result/{date}-{t1}-{t2}.png',dpi=600)
    plt.show()
   #nx.draw(G,pos=nx.spring_layout(G),with_labels=True,node_color=list(partition.values())
   # ,edge_color='grey', node_size=400, alpha=0.5 )
    return G,par,sub_g

def data_by_date(data, date):
    return data.loc[pd.to_datetime(data['date']) == pd.to_datetime(date)]

def get_keywords(G, top, D=None):
    G_tmp = nx.maximum_spanning_tree(G)
    k = min(G.number_of_nodes(), top)
    B = nx.betweenness_centrality(G_tmp, weight='weight')
    D = nx.degree(G, weight='weight') if not D else D
    P = nx.pagerank(G_tmp)
    val_tmp = {i: B[i] * D[i] * (P[i]+0.0001) for i in G.nodes()}
    return ' '.join([i[0] for i in sorted(val_tmp.items(), key=lambda x:x[1], reverse=True)[:k]])

if __name__ == '__main__':
    w,d_range = read('./Data/d1.csv')
    title='1020'
    date='2020-02-01'
    t1=0.001
    t2=0.0001
    w = data_by_date(w,date)
    da = preprocess(w) 
    n = NTDE_4(da,t1,t2)
    G,par,sub_graph = mygraph(n,da,date,t1,t2)
    with open("./result/%s.txt"%title,"a") as f:
            f.write('\n----------------------------\n')
            f.write(f'{date}-{t1}-{t2}\n')
            for subgraph,i in zip(sub_graph,range(len(sub_graph))):
                keywords= get_keywords(subgraph, 10)
                print(keywords)
                f.write(f'com{i}:{keywords}\n')
