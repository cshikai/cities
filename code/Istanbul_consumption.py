#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 01:15:28 2018

@author: kai
"""

from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import pandas as pd 
from glob import glob 
import numpy as np 
import networkx as nx 
import os, json, time, csv
import pickle as pkl 
import matplotlib.pyplot as plt 
from collections import Counter 
import scipy 
from datetime import datetime
from scipy import stats
from geopy.distance import vincenty
import random
import copy 
import math
from statsmodels.formula.api import ols
from geopy.distance import great_circle
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("whitegrid")


from Istanbul_Flow_Regression import Result, normalise_data

#helper function to normalize dataset


def getCSdistance(area1,area2,area2category):
    qk=copy.deepcopy(area2category[area1])
    pk=copy.deepcopy(area2category[area2])
    normq=float(sum(qk.values()))
    normp=float(sum(pk.values()))
    total= list(set(qk.keys()) | set(pk.keys()))
    for key in qk.keys():
        qk[key]=qk[key]/normq
        
    for key in pk.keys():
        pk[key]=pk[key]/normp
    d=0
    for cat in total:
        if cat in qk and cat in pk:
            d+=(qk[cat]-pk[cat])*(qk[cat]-pk[cat])
        elif (cat in qk) and (not cat in pk):
            d+=qk[cat]*qk[cat]
        else:
            d+=pk[cat]*pk[cat]
    return d

def getDivergence(area2distribution,G_cus_area):
    area2divergence={}
    for area,distribution in area2distribution.iteritems():
        area2divergence[area]=np.sum([getCSdistance(area,nei,area2distribution)*G_cus_area[area][nei]['weight'] for nei in  G_cus_area.neighbors(area)])
    return area2divergence
    
def getPairwiseDiff(area2category,G_cus_area):
    area2Pairwise={}
    for area in area2category:
        neighs=G_cus_area.neighbors(area)
        divs=[]
        for i in range(len(neighs)):
            for j in range(i+1,len(neighs)):
                w=G_cus_area[area][neighs[i]]['weight'] *G_cus_area[area][neighs[j]]['weight']
                divs.append(w*getCSdistance(neighs[i],neighs[j],area2category))
        area2Pairwise[area]=np.sum(divs)
    return area2Pairwise

#advanced regression evaluation        
def regressionPlotsArea2(yvar,Xvars,xVarsC,df,cities):
    feature=df
    print '\n================{}=============='.format(yvar)
    
    print feature[yvar].mean()
    
    xC=''
    for v in xVarsC:
        xC=xC +'+'+v

    model = ols("{} ~{} ".format(yvar,xC), feature).fit()
    

    sse = ((model.resid)**2).sum() / ((feature[yvar])**2).sum()
    ess=model.ess/((feature[yvar])**2).sum()
    aic=model.aic
    
    x=Xvars[0]
    for i in range(1,len(Xvars)):
        x=x+'+'+Xvars[i]
    
    print 'The diversity variables are : {}'.format(x)
    model2 = ols("{} ~ {} ".format(yvar,x),feature).fit()
    sse2 = ((model2.resid)**2).sum() / ((feature[yvar])**2).sum()
    ess2=model2.ess/((feature[yvar])**2).sum()
    aic2=model2.aic
    print 'Before SSE : {}'.format(sse)
    print 'After SSE : {}'.format(sse2)
    print 'gain : {}'.format((sse-sse2))
    
    print '\n'
    print 'Before ESS : {}'.format(ess)
    print 'After ESS : {}'.format(ess2)
    print 'gain : {}'.format((sse-sse2))


    print '\n'
    print 'Before AIC : {}'.format(aic)
    print 'After AIC : {}'.format(aic2)
    

### end of helper functions####
    
def getShop2id():
    df=pd.read_csv('Data_For_Kai/1-shopidcustid.txt',names=range(4017))
    shop2id={}
    def a2id(row):
        key=int(row[0])
        values=row.loc[1:]
        values=values.dropna().astype(int)
        shop2id[key]=list(values)
    df.apply(a2id,1)
    return shop2id

def getArea2id(shop2id):
    df=pd.read_csv('Data_For_Kai/5-shopiddistid.csv')
    areaID2shopID={}
    shopID2areaID={}
    def s2a(row):
        shop=row['shop_id']
        area=row['district_id']
        shopID2areaID[shop]=area
        if area not in areaID2shopID:
            areaID2shopID[area]=[]
        areaID2shopID[area].append(shop)
    df.apply(s2a,1)
    area2id={}
    for key,value in shop2id.iteritems():
        area=shopID2areaID[key]
        if area not in area2id:
            area2id[area]=[]
        area2id[area]=area2id[area]+value
    return area2id,shopID2areaID,areaID2shopID

def getArea2idW():
    df=pd.read_csv('Data_For_Kai/7-custidwdistid.csv',index_col='customer_id')
    Dict={}
    for index,row in df.iterrows():
        area=str(row['wdistrict_id'])
        if not area in Dict:
            Dict[area]=[]
        Dict[area].append(index)
    return Dict

def getArea2idH():
    df=pd.read_csv('Data_For_Kai/6-custidhdistid.csv',index_col='customer_id')
    Dict={}
    for index,row in df.iterrows():
        area=str(row['hdistrict_id'])
        if not area in Dict:
            Dict[area]=[]
        Dict[area].append(index)
    return Dict
         
def buildAreaGraph(graph_type,area2id):
    if graph_type=='customer':
        G = nx.Graph()
        for area in area2id:
            G.add_node(str(area))
        for i  in range(len(area2id)):
            for j in range(i+1,len(area2id)):
                a1=area2id.keys()[i]
                a2=area2id.keys()[j]
                s1 = set(area2id[a1])
                s2 = set(area2id[a2])
                if len(s1) > 0 and len(s2) > 0:
                    inter = len(s1 & s2)
                    union = len(s1) + len(s2) - inter
                    if inter > 0:
                        G.add_edge(str(a1), str(a2), weight=1.0*inter/union)
        return G
    elif graph_type == 'dist':
        df=pd.read_csv('Data_For_Kai/2-districtijdistance.csv',names=range(3))
        G_dist=nx.Graph()
        thres=100000
        def getDistance(i,j):
            return df[(df[0]==i) & (df[1]==j)][2].iloc[0]
        nodes=set((df[0].unique())) | set((df[1].unique()))
        nodes=np.sort(list(nodes))
        for node in nodes:
            G_dist.add_node(str(node))
        for i  in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                u=nodes[i]
                v=nodes[j]
                distance = getDistance(u,v)
                if distance<thres:
                    G_dist.add_edge(str(u),str(v),weight=1/distance)
        return G_dist

def getUser2shop(shop2id):
    '''
        this function returns the stores each user visited 
    '''
    user2shop = {}
    for k in shop2id:
        for j in shop2id[k]:
            if j in user2shop:
                user2shop[j].append(k)
            else:
                user2shop[j] = [k]
    return user2shop
    
def areaGraphConstruction():
    shop2id=getShop2id()
    user2shop=getUser2shop(shop2id)
    shoparea2id,shopID2areaID,areaID2shopID=getArea2id(shop2id)
    shop_G_cus_area=buildAreaGraph('customer',shoparea2id)
    shop_G_dist_area=buildAreaGraph('dist',shoparea2id)
    json.dump(shop2id, open('istanbulShop2id.json', 'wb'))
    json.dump(user2shop, open('istanbulUser2shop.json', 'wb'))
    json.dump(shoparea2id, open('istanbulArea2id.json', 'wb'))
    json.dump(areaID2shopID, open('istanbulAreaID2shopID.json', 'wb'))
    json.dump(shopID2areaID, open('istanbulShopID2areaID.json', 'wb'))
    nx.write_gpickle(shop_G_cus_area,'shopistanbulG_cus_area.pkl')
    nx.write_gpickle(shop_G_dist_area,'shopistanbulG_dist_area.pkl')



def getControls(area2id,areaID2shopID):
    numCus,numShop={},{}
    for area in area2id.keys():
        numCus[area]=len(area2id[area])
        numShop[area]=len(areaID2shopID[area])
    return numCus,numShop        

def getCentralities(G_cus_area,G_dist_area):
    degree_cus,degree_dist={},{}
    areas=G_cus_area.nodes()
    for area in areas:
        degree_cus[area]=np.sum([G_cus_area[area][i]['weight'] for i in G_cus_area.neighbors(area)])
        degree_dist[area]= np.sum([G_dist_area[area][i]['weight'] for i in G_dist_area.neighbors(area)])
    eigen_cus = nx.eigenvector_centrality(G_cus_area,max_iter=100,tol=0.00001)
    eigen_dist = nx.eigenvector_centrality(G_dist_area,max_iter=100,tol=0.00001)
    return degree_cus,degree_dist,eigen_cus,eigen_dist






def getControlsHome(area2idH):
    numHome={}
    for area in area2idH.keys():
        numHome[area]=len(area2idH[area])
    return numHome    


def areaAddGraphFeatures(area2numCus,area2numShop,area2DegreeCus,area2DegreeDist,area2EigenCus,area2EigenDist,areaIndptVars):
    
    #controls
    numberOfCustomers=[]
    numberOfShops=[]
    
    #Shop properties
    #network
   
    weightedDegreeCentralityCus=[]
    weightedDegreeCentralityDist=[]
    weightedEigenCentralityCus=[]
    weightedEigenCentralityDist=[]

    ids=area2numCus.keys() 
    for area in area2numCus.keys():
        numberOfCustomers.append(area2numCus[area])
        numberOfShops.append(area2numShop[area])
        
        weightedDegreeCentralityCus.append(area2DegreeCus[area])
        weightedDegreeCentralityDist.append(area2DegreeDist[area])
        weightedEigenCentralityCus.append(area2EigenCus[area])
        weightedEigenCentralityDist.append(area2EigenDist[area])
    
        
    featNames=['district_id']
    featVals=[ids]
   
      
    featVals=[numberOfCustomers
              ,numberOfShops
              ,weightedDegreeCentralityCus
              ,weightedDegreeCentralityDist
    ,weightedEigenCentralityCus
    ,weightedEigenCentralityDist]+featVals
    
    featNames=['numberOfCustomers'
              ,'numberOfShops'
              ,'weightedDegreeCentralityCus'
              ,'weightedDegreeCentralityDist'
    ,'weightedEigenCentralityCus'
    ,'weightedEigenCentralityDist']+featNames
    
    finalDf=pd.DataFrame(np.stack(featVals,1),
                       columns=(featNames))
    finalDf=pd.merge(finalDf,areaIndptVars,how='left',on='district_id')
    return finalDf

#netowrk features , controls, and indepdenant variables
def areaFeatureExtraction():
    shopID2areaID=json.load(open('istanbulShopID2areaID.json','rb'))
    areaID2shopID=json.load(open('istanbulAreaID2shopID.json','rb'))
    area2id=json.load(open('istanbulArea2id.json','rb'))
    user2shop=json.load(open('istanbulUser2shop.json','rb'))
    G_cus_area=nx.read_gpickle('shopistanbulG_cus_area.pkl')
    G_dist_area=nx.read_gpickle('shopistanbulG_dist_area.pkl')

    area2numCus,area2numShop=getControls(area2id,areaID2shopID)
    
    area2DegreeCus,area2DegreeDist,area2EigenCus,area2EigenDist=getCentralities(G_cus_area,G_dist_area)

    areaIndptVars=getIndptVars(shopID2areaID)
    areaIndptVars['district_id']= areaIndptVars['district_id'].astype(str)
    df=areaAddGraphFeatures(area2numCus,area2numShop,area2DegreeCus,area2DegreeDist,area2EigenCus,area2EigenDist ,areaIndptVars)
    df.to_csv('istanbulAreaDataWithAllFeatures.csv',encoding='utf-8',sep='\t',index=False)

#helper for housing prices

    

#helper for consumption diversity 
def getConsumptionDiversity():
    def getEntropy(chunk):
        return scipy.stats.entropy(chunk.counts.values)    
    raw=pd.read_csv('istanbulData/raw/SU_ORNEKLEM_KK_HAR_BILGI.txt')
    raw=raw[(raw['UYEISYERI_ID_MASK']!=999999) & (raw.ONLINE_ISLEM ==0)]
    raw.rename(columns={'MUSTERI_ID_MASK':'customer_id','UYEISYERI_ID_MASK':'shop_id'},inplace=True)
    df=pd.read_csv('istanbulData/raw/5-shopiddistid.csv')
    df.rename(columns={'district_id':'sdistrict_id'},inplace=True)
    raw=pd.merge(raw,df,on='shop_id',how='left')
    raw.dropna(subset=['sdistrict_id'],inplace=True)
    df=pd.read_csv('istanbulData/raw/4-shopid_mccmerged.csv')
    raw=pd.merge(raw,df,on='shop_id',how='left')
    raw.dropna(subset=['mcc_detailed'],inplace=True)
    dDf=pd.DataFrame('',index=raw.sdistrict_id.unique(),columns=list())
    dDf.index.name='sdistrict_id'
    dfg=raw.groupby(['mcc_merged','sdistrict_id']).size()
    dfg = pd.DataFrame(dfg, columns=['counts'])
    dfg.reset_index(inplace=True)
    dDf['UA']=dfg.groupby('sdistrict_id').apply(getEntropy)
    dDf['district_id']=dDf.index
    #dDf.to_csv('istanbulData/UA.csv',index=False)
    return dDf



#run to merge datasets and get regression dataset
def readDataIstanbul():
    df=pd.read_csv('istanbulData/istanbulAreaDataWithAllFeatures.csv',sep='\t')
    df.dropna(subset=['delta1416p'],inplace=True)
    
    df2=getPopulationDense()
    df=pd.merge(df,df2,on='district_id',how='left')
    
    df2= getTurkHPIWage()
    df=pd.merge(df,df2,on='district_id',how='left')
    df['logpopdse']=np.log(df['popdse'])
    
    df['log2015dse']=np.log(df['2015']/df.area)
    df['log2015']=np.log(df['2015'])

    df2=getConsumptionDiversity()
    df=pd.merge(df,df2,on='district_id',how='left')
    df=df[['UA',u'popdse','weightedEigenCentralityDist','HPI','district_id','district_name','delta1416p']]
    df.HPI=df.HPI/(df.HPI.min())
    df.to_csv('istanbul_cc.csv',index=False)
    return df

#helper for aggregated demographics
def agg(chunk):
        inco=chunk.income.mean()
        ag=chunk.age.mean()
        gen=(chunk.gender=='M').sum()*1.0 / len(chunk)
        mari=(chunk.marital=='MARRIED').sum()*1.0 / len(chunk)
        high=['UNIVERSITY','COLLEGE','MASTERS','PHD']
        edu=(chunk.education.isin(high).sum())*1.0/len(chunk)
        idx=chunk.hdistrict_id.values[1]
        return [inco,ag,gen,mari,edu,idx]
    

def ConsumptionAndGrowth(Result):
    name = 'consumption_and_growth'
    
    def __init__(self):
        df = pd.read_csv('../data/istanbul/istanbul_cc.csv')
        df.rename(columns={'weightedEigenCentralityDist': 'geo_centrality'},inplace=True)
    
        self.df['istanbul'] = normalise_data(df,omit=['district_id','district_name')
        
        df = pd.read_csv('../data/beijing/beijing_cc.csv')
        df.rename(columns={'economy':'delta1416p','dEign':'geo_centrality','categoryEntropy':'UA'},inplace=True)
        self.df['beijing'] = normalize_data(df,omit=['district_id'])
        
        df = pd.read_csv('../data/usa/usa_cc.csv')
        #deltap or deltacp
        df.rename(columns={'ua':'UA','geo':'geo_centrality','deltap':'delta1416p'},inplace=True)
        
    def get_regressions(self):
        df_indexes = ['istanbul']
        cols = ['UA','popdse','HPI','geo_centrality']
        combinations = [(0,1,2,3)]
        ys = ['delta1416p']
        for y in ys:
            self.run_regressions(df_indexes,cols,combinations,y)
# =============================================================================
# if __name__== "__main__":
#     #newdf=pd.DataFrame()    
#     #newdf['income'],newdf['age'],newdf['gender'],newdf['married'],newdf['edu'],newdf['district_id']=zip(*demog.groupby('hdistrict_id').apply(agg))      
# 
#     df=pd.read_csv('istanbul_cc.csv')
#     df_norm=normalise_data(df)
#     df_norm.to_csv('yelp/istanbul_cc_n.csv',index=False)
#     areaGraphConstruction()
#     areaFeatureExtraction()
#     areaRegression()
#        
#     pass
# =============================================================================
