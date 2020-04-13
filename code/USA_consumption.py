#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 23:18:41 2017

@author: kai
"""
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

#%%

#==============================================================================
# tweets = []
# for line in open('/Users/kai/Desktop/yelp/yelp_dataset/dataset/tip.json','rb'):
#     tweets.append(json.loads(line))
#     
# #each review has a 'date'
#==============================================================================
#%%
def getTotalConsumption():
    consume={}
    with open('../diversityData/yelp/round10/review.json','rb') as f:
        for line in open('../diversityData/yelp/round10/review.json','rb'):
            t=json.loads(line)
            shop=t['business_id']
            if shop not in consume:
                consume[shop]=0
            consume[shop]+=1

    with open('../diversityData/yelp/round10/tip.json','rb') as f:
        for line in open('../diversityData/yelp/round10/tip.json','rb'):
            t=json.loads(line)
            shop=t['business_id']
            if shop not in consume:
                consume[shop]=0
            consume[shop]+=1  
    return consume

def getYearlyConsumption():
    consumebyyear={}
    with open('../diversityData/yelp/round10/review.json','rb') as f:
        for line in f:
            t=json.loads(line)
            year=int(t['date'].split('-')[0])
            prevyear=year-1
            try:
                yeardic=consumebyyear[year]
            except:
                consumebyyear[year]={}
                yeardic=consumebyyear[year]
            try:
                prevyeardic=consumebyyear[prevyear]
            except:
                consumebyyear[prevyear]={}
                prevyeardic=consumebyyear[prevyear]
                
            shop=t['business_id']
            if shop not in yeardic:
                yeardic[shop]=0
            yeardic[shop]+=1
            if shop not in prevyeardic:
                prevyeardic[shop]=0
            prevyeardic[shop]+=1
    with open('../diversityData/yelp/round10/tip.json','rb') as f:
        for line in f:
            t=json.loads(line)
            year=int(t['date'].split('-')[0])
            prevyear=year-1
            try:
                yeardic=consumebyyear[year]
            except:
                consumebyyear[year]={}
                yeardic=consumebyyear[year]
            try:
                prevyeardic=consumebyyear[prevyear]
            except:
                consumebyyear[prevyear]={}
                prevyeardic=consumebyyear[prevyear]
                
            shop=t['business_id']
            if shop not in yeardic:
                yeardic[shop]=0
            yeardic[shop]+=1
            if shop not in prevyeardic:
                prevyeardic[shop]=0
            prevyeardic[shop]+=1
    return consumebyyear

def getArea2ConsumptionDiv(newshop):
    shops=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    df=pd.DataFrame.from_dict(shops,orient='index')
    shoplist=list(df.index)
    area2category = {}
    for shopname,count in newshop.iteritems():
        if shopname in shoplist:
            cat = df.loc[shopname,'categories']
            area=df.loc[shopname,'GEOID']
            if area not in area2category:
                area2category[area]={}
            if cat not in area2category[area]:
                 area2category[area][cat]=0
            area2category[area][cat]+=count
    area2entropy = {}
    for k in area2category:
        area2entropy[k] = scipy.stats.entropy(np.array(area2category[k].values()) )
    return area2entropy

def getConsumptionDiversity():
    #total consumption
    shops=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    df=pd.DataFrame.from_dict(shops,orient='index')
    areas=df.GEOID.unique()
    df=pd.DataFrame()
    df['GEOID']=areas
    a2e=getArea2ConsumptionDiv(consume)
    df2=pd.DataFrame(np.stack([a2e.keys(),a2e.values()],1),columns=['GEOID','UA'])
    df=pd.merge(df,df2,on='GEOID',how='left')
    df.to_csv('UA.csv',index=False)


    #yearly consumption
    shops=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    df=pd.DataFrame.from_dict(shops,orient='index')
    areas=df.GEOID.unique()
    df=pd.DataFrame()
    df['GEOID']=areas
    for year in consumebyyear.keys():
        a2e=getArea2ConsumptionDiv(consumebyyear[year])
        df2=pd.DataFrame(np.stack([a2e.keys(),a2e.values()],1),columns=['GEOID',str(year)+'UA'])
        df=pd.merge(df,df2,on='GEOID',how='left')
    df.to_csv('UA2.csv',index=False)

#dictonaries for housekeeping
def getArea2id(shop,shop2id):
    area2id={}
    areaID2shopID={}
    shopID2areaID={}
    num2name={}
    for key,value in shop2id.iteritems():
        area=shop[key]['GEOID']
        if area not in area2id:
            area2id[area]=[]
            areaID2shopID[area]=[]
            num2name[area]=shop[key]['NAME']
        areaID2shopID[area].append(key)
        shopID2areaID[key]=area
        area2id[area]=area2id[area]+value
    return area2id,num2name,shopID2areaID,areaID2shopID

#get lat lons of shops , both indiviudally and also groups of shops in a census block
def getAreaLatLon(shop,areaID2shopID):
    shops=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    df=pd.DataFrame.from_dict(shops,orient='index')
    area2location = {}
    shop2location = {}
    for index,row in df.iterrows():
        x = row['lat']
        y = row['lon']
        shop2location[row['bizID']] = (float(x), float(y))
    for area in areaID2shopID:
        if area not in area2location:
                area2location[area] = []
        for business in areaID2shopID[area]:
            area2location[area].append(shop2location[business])
    return area2location,shop2location

#build networks based on shared customers and based on the geographical distance between each area's hotspot
def buildAreaGraph(graph_type,area2id,area2location,num2name):
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
        G_dist=nx.Graph()
        thres=99999999
        area2central = {}
        for node in area2location:
            x = np.mean([xx[0] for xx in area2location[node]])
            y = np.mean([xx[1] for xx in area2location[node]])
            area2central[node] = (x,y)
            G_dist.add_node(str(node))
        for i  in range(len(area2central)):
            for j in range(i+1,len(area2central)):
                u=area2central.keys()[i]
                v=area2central.keys()[j]
                x1 = area2central[u][0]
                y1 = area2central[u][1]
                x2 = area2central[v][0]
                y2 = area2central[v][1]
                distance = vincenty((x1, y1), (x2, y2)).kilometers
                if distance<thres:
                    G_dist.add_edge(str(u),str(v),weight=1/distance)
        return G_dist

#helper function, ran to generate graph as pkl file.
def areaGraphConstruction():
    shop=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    shop2id=json.load(open('../diversityData/yelp/round10/yelpShop2id.json','rb'))
    area2id,num2name,shopID2areaID,areaID2shopID=getArea2id(shop,shop2id)
    area2location,shop2location=getAreaGPS(shop,areaID2shopID)
    G_cus_area=buildAreaGraph('customer',area2id,area2location,num2name)
#==============================================================================
#     G_dist_area=buildAreaGraph('dist',area2id,area2location,num2name)
#==============================================================================
    
    json.dump(area2id, open('../diversityData/yelp/round10/yelpArea2id.json', 'wb'))
    json.dump(areaID2shopID, open('../diversityData/yelp/round10/yelpAreaID2shopID.json', 'wb'))
    json.dump(shopID2areaID, open('../diversityData/yelp/round10/yelpShopID2areaID.json', 'wb'))
    nx.write_gpickle(G_cus_area,'../diversityData/yelp/round10/yelpG_cus_area.pkl')
    nx.write_gpickle(G_dist_area,'../diversityData/yelp/round10//yelpG_dist_area.pkl')


#%%

#diversity of shops AVAILABLE in the region (i.e not weighted by consumption frequency)
def getArea2AvailablityDiversity(areaID2shopID,shop):
    shops=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    df=pd.DataFrame.from_dict(shops,orient='index')
    shop2category = {}
    for index,row in df.iterrows():
        mid=row['bizID']
        shop2category[mid] = row['categories']
    area2category = {}
    for areaID in areaID2shopID:
        area2category[areaID] = {}
        for shopID in areaID2shopID[areaID]:
            c = shop2category[shopID]
            if c not in area2category[areaID]:
                area2category[areaID][c] = 1
            else:
                area2category[areaID][c] += 1
    area2entropy = {}
    for k in area2category:
        area2entropy[k] = scipy.stats.entropy(np.array(area2category[k].values()) / 1.0 / np.sum(area2category[k].values()))
    return area2category,area2entropy


#diversity of shops consumed in the region (i.e weighted by number of people who visted each shop , Note: not the same as consumption , as the same person can visit multiple times)
def getArea2PeopleVisitDiversity(areaID2shopID,shop):
    shops=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    shop2id=json.load(open('../diversityData/yelp/round10/yelpShop2id.json','rb'))
    
    df=pd.DataFrame.from_dict(shops,orient='index')
    shop2category = {}
    for index,row in df.iterrows():
        mid=row['bizID']
        shop2category[mid] = row['categories']
    area2category = {}
    for areaID in areaID2shopID:
        area2category[areaID] = {}
        for shopID in areaID2shopID[areaID]:
            c = shop2category[shopID]
            if c not in area2category[areaID]:
                area2category[areaID][c] = len(shop2id[shopID])
            else:
                area2category[areaID][c] += len(shop2id[shopID])
    area2entropy = {}
    for k in area2category:
        area2entropy[k] = scipy.stats.entropy(np.array(area2category[k].values()) / 1.0 / np.sum(area2category[k].values()))
    return area2category,area2entropy


#get the diversity of customer's income, estimated based on expenditures by each customer on yelp
def getCustomerIncomeEntropy(area2id,user2shop,shop):
    shops=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    incomeData=pd.DataFrame.from_dict(shops,orient='index')
    incomeData=incomeData[incomeData['price'].notnull()]
    priceDict={}
    for index, row in incomeData.iterrows():
        priceDict[row['bizID']]=row['price']
    userIncome=[]
    userID=[]
    emptyID=[]
    for key,value in user2shop.iteritems():
        summ=0.0
        count=0
        for shop in list(value):
            if shop in priceDict:
                summ+=priceDict[shop]
                count+=1
        if count!=0:
            userIncome.append(summ/count)
            userID.append(key)
        else:
            emptyID.append(key)
    category=pd.qcut(userIncome,4,labels=False,duplicates='drop').tolist()
    incomeDic={}
    for i in range(len(category)):
        incomeDic[userID[i]]=category[i]
    for i in emptyID:
        incomeDic[i]='NoInfo'
    IncomeEntropyID=[]
    IncomeEntropy=[]
    Num_customers=[]
    for area,idx in area2id.iteritems():
        count_dict = {}
        for name in idx:
            if incomeDic[name] not in count_dict:
                count_dict[incomeDic[name]] = 1
            else:
                count_dict[incomeDic[name]] += 1
        cnt = [1.0 * i/sum(count_dict.values()) for i in count_dict.values()]
        entropy = 0#sc.stats.entropy(cnt)
        for e in cnt: 
            entropy -= e * np.log(e)
        IncomeEntropy.append(entropy)
        IncomeEntropyID.append(area)
        Num_customers.append(len(idx))
    area2income={}
    area2numCus={}
    for i in range(len(IncomeEntropyID)):
        area2income[IncomeEntropyID[i]]=IncomeEntropy[i]
        area2numCus[IncomeEntropyID[i]]=Num_customers[i]
    return area2income,area2numCus


#get the average diversity of the categories shops that customers in the region visits
def getAreaCustomerEntropy(user2shop,shop,area2id):
    cusEntro={}
    for key,value in user2shop.iteritems():
        entro={}
        for v in value:
            shopvcat=shop[v]['categories']
            if shopvcat not in entro :
                entro[shopvcat]=1
            else:
                entro[shopvcat]+=1
        cnt = [1.0 * i/sum(entro.values()) for i in entro.values()]
        entropy = 0#sc.stats.entropy(cnt)
        for e in cnt: 
            entropy -= e * np.log(e)
        cusEntro[key]=entropy
    areaCusEntropy={}
    for area,value in area2id.iteritems():
        areaCusEntropy[area]=np.mean([cusEntro[name] for name in value])
    return areaCusEntropy

#helper to calculate distance between probabilities
def getDivergence(area1,area2,area2category):
    qk=copy.deepcopy(area2category[area1])
    pk=copy.deepcopy(area2category[area2])
    normq=float(sum(qk.values()))
    normp=float(sum(pk.values()))
    total= list(set(qk.keys()) | set(pk.keys()))
    
    for key,value in qk.iteritems():
        qk[key]=value/normq
        
    for key,value in pk.iteritems():
        pk[key]=value/normp
        
    d=0
    for cat in total:
        if cat in qk and cat in pk:
            d+=(qk[cat]-pk[cat])*(qk[cat]-pk[cat])
        elif (cat in qk) and (not cat in pk):
            d+=qk[cat]*qk[cat]
        else:
            d+=pk[cat]*pk[cat]
    return d

#helper 
def getPairWiseDiff(area2category,G_cus_area):
    area2Pairwise={}
    for area in area2category:
        neighs=G_cus_area.neighbors(area)
        divs=[]
        for i in range(len(neighs)):
            for j in range(i+1,len(neighs)):
                divs.append(getDivergence(neighs[i],neighs[j],area2category))
        area2Pairwise[area]=np.mean(divs)
    return area2Pairwise


#helper to generate indicators about the region's macro state
def getIndptVars(shop,shopID2areaID):
    df2=pd.read_csv('../diversityData/yelp/round10/income_and_population.csv')
    df2=df2.rename(columns = {'GeoFIPS':'GEOID'})
    shop=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    df=pd.DataFrame.from_dict(shop,orient='index')
    area2rate={}
    area2review={}
    grp=df.groupby(['GEOID'])
    for index,group in grp:
        area2rate[str(index)]=group['rating'].mean()
        area2review[str(index)]=group['numReview'].sum()
    return df2,area2rate,area2review

#helper tool to normalize data
def norm(col):
    col2=col-np.mean(col)
    col2=col2/np.std(col2)
    return col2

#helper to generate features
def areaAddGraphFeatures(dfInd,area2category2,area2entropy2,area2entropy,G_cus_area,G_dist_area,area2review,area2rating,area2numCus,area2income,areaCusEntropy,areaID2shopID,area2Pairwise,area2category):
    wentro=[]
    reviews=[]
    ratings=[]
    entro=[]
    diverge=[]
    degreeCus=[]
    degreeDist=[]
    numCus=[]
    incomeEntro=[]
    avgCusEntro=[]
    numShops=[]
    Pairwise=[]
    ids=area2entropy.keys()    
    for key in area2entropy:
        if len(G_cus_area.neighbors(key)) == 0:
           reviews.append(area2review[key])
           ratings.append(area2rating[key])
           entro.append(area2entropy[key])
           diverge.append(None)
           degreeCus.append(None)
           degreeDist.append(None)
           numCus.append(area2numCus[key])
           incomeEntro.append(area2income[key])
           avgCusEntro.append(None)
           numShops.append(0)
           Pairwise.append(0)
           wentro.append(area2entropy2[key])
        else:
            reviews.append(area2review[key])
            ratings.append(area2rating[key])
            entro.append(area2entropy[key])
            diverge.append(np.mean([getDivergence(key,nei,area2category) for nei in  G_cus_area.neighbors(key)]))
            degreeCus.append(sum([G_cus_area[key][i]['weight'] for i in G_cus_area.neighbors(key)]))
            degreeDist.append(sum([G_dist_area[key][i]['weight'] for i in G_dist_area.neighbors(key)]))
            numCus.append(area2numCus[key])
            incomeEntro.append(area2income[key])
            avgCusEntro.append(areaCusEntropy[key])
            numShops.append(len(areaID2shopID[key]))
            Pairwise.append(area2Pairwise[key])
            wentro.append(area2entropy2[key])
    finalDf=pd.DataFrame(np.stack([ids,wentro,reviews,ratings,entro,diverge,degreeCus,degreeDist,numCus,numShops,incomeEntro,avgCusEntro,Pairwise],1),
                       columns=['GEOID','weightedcategoryEntropy','totalReviews','averageRating','categoryEntropy','averageCategoryDivergence','degreeCus','degreeDist','numberofCustomers','numberofShops','incomeEntropy',
                                'avgCusEntropy','pairwise'])
    print 'computing eigen for cus net'
    centrality_cus = nx.eigenvector_centrality(G_cus_area,max_iter=100,tol=0.00001)
    print 'computing eigen for dist net'
    centrality_dist = nx.eigenvector_centrality(G_dist_area,max_iter=100,tol=0.00001)
    cusKey=centrality_cus.keys()
    cusVal=centrality_cus.values()
    dKey=centrality_dist.keys()
    dVal=centrality_dist.values()
    df1=pd.DataFrame(np.stack([cusKey,cusVal],1),columns=['GEOID','cusEigen'])
    df2=pd.DataFrame(np.stack([dKey,dVal],1),columns=['GEOID','dEigen'])
    addDf=pd.merge(df1,df2,how='left',on='GEOID')
    finalDf=pd.merge(finalDf,addDf,how='left',on='GEOID')
    finalDf=pd.merge(finalDf,dfInd,how='left',on='GEOID')
    return finalDf

#helper tool, ran to generate .csv with all the features and independant variables
def areaFeatureExtraction():
    shopID2areaID=json.load(open('../diversityData/yelp/round10/yelpShopID2areaID.json','rb'))
    shop=json.load(open('../diversityData/yelp/round10/yelpShop.json','rb'))
    areaID2shopID=json.load(open('../diversityData/yelp/round10/yelpAreaID2shopID.json','rb'))
    area2id=json.load(open('../diversityData/yelp/round10/yelpArea2id.json','rb'))
    user2shop=json.load(open('../diversityData/yelp/round10/yelpUser2shop.json','rb'))
    
    area2category,area2entropy=getArea2AvailablityDiversity(areaID2shopID,shop)
    area2category2,area2entropy2=getArea2PeopleVisitDiversity(areaID2shopID,shop)
    area2income,area2numCus=getCustomerIncomeEntropy(area2id,user2shop,shop)
    
    areaCusEntropy=getAreaCustomerEntropy(user2shop,shop,area2id)
    G_cus_area=nx.read_gpickle('../diversityData/yelp/round10/yelpG_cus_area.pkl')
    G_dist_area=nx.read_gpickle('../diversityData/yelp/round10/yelpG_dist_area.pkl')
    area2Pairwise=getPairWiseDiff(area2category,G_cus_area)
    df2,area2rating,area2review=getIndptVars(shop,shopID2areaID)
    df2['GEOID']=df2['GEOID'].astype(str)
    df=areaAddGraphFeatures(df2,area2category2,area2entropy2,area2entropy,G_cus_area,G_dist_area,area2review,area2rating,area2numCus,area2income,areaCusEntropy,areaID2shopID,area2Pairwise,area2category)
    df.to_csv('../diversityData/yelp/round10/yelpAreaDataWithAllFeatures.csv',encoding='utf-8',sep='\t',index=False)
        




def addControlVariablesUS():
    df=pd.read_csv('yelp/yelpAreaDataWithAllFeatures.csv',sep='\t')
    popDf=getPop
    years=[2010,2011,2012,2013,2014,2015]
    for year in years[1:]:
        df['delta'+str(year)]=df['income'+str(year)]-df['income'+str(year-1)]
        df['delta'+str(year)+'p']=df['delta'+str(year)]/df['income'+str(year-1)]
        df['delta'+str(year)+'pc']=df['incomePer'+str(year)]-df['incomePer'+str(year-1)]
        df['delta'+str(year)+'pcp']=df['delta'+str(year)+'pc']/df['incomePer'+str(year-1)]
    df.index=df.GEOID
    ua1=pd.read_csv('yelp/UA.csv')
    ua1.index=ua1.GEOID
    ua2=pd.read_csv('yelp/UA2.csv')
    ua2.index=ua2.GEOID
    GEOIDs=df.GEOID.tolist()
    df2=pd.read_csv('yelp/CensusData/HPI.csv')
    df2.rename(columns={'FIPS code':'GEOID','HPI with 2000 base':'HPI2'},inplace=True)
    df2.index=df2.Year.astype(str)+'_'+df2.GEOID.astype(str)
    deltaps,deltapcps,names,yearss,geoids,HPIs,geos,uas,pops=[],[],[],[],[],[],[],[],[]
    for year in years[1:]:
        for geoid in GEOIDs:
            uaa=ua2.loc[geoid,str(year-1)+'UA']
            if uaa !=0 and not (np.isnan(uaa)):
                uas.append(uaa)
                names.append(str(geoid)+'_'+str(year))
                yearss.append(year)
                geoids.append(geoid)
                deltaps.append(df.loc[geoid,'delta'+str(year)+'p'])
                deltapcps.append(df.loc[geoid,'delta'+str(year)+'pcp'])
                HPIs.append(df2.loc[str(year)+'_'+str(geoid),'HPI2'])
                geos.append(df.loc[geoid,'dEigen'])
                pops.append(popDf.loc[geoid,'pop'+str(year)])            
    ddf=pd.DataFrame(np.stack([deltaps,deltapcps,names,yearss,geoids,HPIs,geos,uas,pops],1),columns=['deltap','deltapcp','name','year','geoid','HPI','geo','ua','popu']) 
    ddf['data_id']=range(len(ddf))
    df3=pd.read_csv('yelp/CensusData/area.csv')
    df3=df3[['GEOID','area']]
    df3.rename(columns={'GEOID':'geoid'},inplace=True)
    ddf=pd.merge(ddf,df3,on='geoid',how='left')
    ddf.popu=ddf.popu/1000
    ddf.area=ddf.area/2.58999
    ddf['popdse']=ddf.popu/ddf.area
    ddf['logpop']=np.log(ddf.popu)
    ddf['logpopdse']=np.log(ddf.popdse)
    ddf.HPI=ddf.HPI/100
    ddf.to_csv('yelp/usa_cc.csv',index=False) 
    return ddf

#generate plot in paper
def plot():
    feature=pd.read_csv('yelp/usa_cc.csv')
    xVarsC=[u'popdse','geo','HPI']
    xVar='ua'
    yVar='deltap'
    
    plt.figure()
    ylab='G : $\Delta$Income'
    xlab='Local goods and service consumption diversity, H'

    plt.subplot(111)
    plt.scatter(feature[xVar].values, feature[yVar].values,color='#4B2750')

    x=feature[xVar].values
    y=feature[yVar].values
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

    plt.xlabel(xlab,size=15)
    plt.ylabel(ylab,size=15)
    ans=feature[[xVar, yVar]].corr()[yVar][xVar]
    plt.text(1.4,0.15,'Correlation : {}'.format(ans),size=15)
    
    plt.savefig('yelp/usa1.png')

    xC=''
    plt.figure()
    plt.subplot(111)
    for v in xVarsC:
        xC=xC +'+'+v
    controlmodel = ols("{} ~ {} ".format(yVar,xC),feature).fit()
    feature['{}_resid'.format(yVar)] = controlmodel.resid
    feature['{}_resid'.format(yVar)] = feature['{}_resid'.format(yVar)] - feature['{}_resid'.format(yVar)].mean()
    feature['{}_resid'.format(yVar)] = feature['{}_resid'.format(yVar)]/ np.std(feature['{}_resid'.format(yVar)])
    controlmodel2 = ols("{} ~ {} ".format(xVar,xC), feature).fit()
    feature['{}_resid'.format(xVar)] = controlmodel2.resid
    feature['{}_resid'.format(xVar)] = feature['{}_resid'.format(xVar)] - feature['{}_resid'.format(xVar)].mean()
    feature['{}_resid'.format(xVar)] = feature['{}_resid'.format(xVar)]/ np.std(feature['{}_resid'.format(xVar)])
    
    plt.scatter(feature[xVar + '_resid'].values, feature[yVar+'_resid'].values, color='#4B2750')
    plt.ylabel('$\Delta$Income residual, after control' ,size=15)
    plt.xlabel('H residual, after control',size=15)
    x=feature[xVar + '_resid'].values
    y=feature[yVar+'_resid'].values
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    
    ans=feature[[xVar+'_resid', yVar+'_resid']].corr()[yVar+'_resid'][xVar+'_resid']
    plt.text(-2,2.5,'Partial Correlation : {}'.format(ans),size=15)
    plt.savefig('yelp/usa2.png')


#helper function -  old exploratory analysis , not in paper 
def regressionPlotsArea(yvar,Xvars,df):
    for i in range(len(Xvars)):
        fet_int = Xvars[i]
        plt.figure(num=None, figsize=(10, 5), dpi=80, edgecolor='k')
        feature=df
        plt.subplot(121)
        plt.scatter(feature[fet_int].values, feature[yvar].values, alpha = 0.3)
        plt.ylabel(yvar)
        plt.xlabel('{}'.format(fet_int))
        plt.title('{} vs. {}'.format(yvar,fet_int))        
        model = ols("{} ~ totalReviews+degreeDist+numberofShops+ numberofCustomers ".format(yvar), feature).fit()
        feature['{}_resid'.format(yvar)] = model.resid
        feature['{}_resid'.format(yvar)] = feature['{}_resid'.format(yvar)] - feature['{}_resid'.format(yvar)].mean()
        feature['{}_resid'.format(yvar)] = feature['{}_resid'.format(yvar)]/ np.std(feature['{}_resid'.format(yvar)])
        model = ols("{}_resid ~  {}".format(yvar,fet_int), feature).fit()
        model1 = ols("{} ~ totalReviews+degreeDist+numberofShops+ numberofCustomers ".format(fet_int), feature).fit()
        
        feature['{}_resid'.format(yvar)] = model.resid
        feature['{}_resid'.format(yvar)] = feature['{}_resid'.format(yvar)] - feature['{}_resid'.format(yvar)].mean()
        feature['{}_resid'.format(yvar)] = feature['{}_resid'.format(yvar)]/ np.std(feature['{}_resid'.format(yvar)])
        
        
        feature['{}_resid'.format(fet_int)] = model1.resid
        feature['{}_resid'.format(fet_int)] = feature['{}_resid'.format(fet_int)] - feature['{}_resid'.format(fet_int)].mean()
        feature['{}_resid'.format(fet_int)] = feature['{}_resid'.format(fet_int)]/ np.std(feature['{}_resid'.format(fet_int)])
        
        
        plt.subplot(122)
        
        plt.scatter(feature[fet_int + '_resid'].values, feature['{}_resid'.format(yvar)].values, alpha = 0.3)
        plt.ylabel('residuals after controlling')
        plt.xlabel('{}'.format(fet_int))
        plt.title('{} vs. {}'.format(yvar,fet_int))
        plt.savefig('../diversityData/yelp/round10/areaPlots/{}/{}_{}.pdf'.format(yvar,yvar,fet_int))
        print feature[[fet_int, '{}_resid'.format(yvar)]].corr()
        print(model.summary())

#helper function -  old exploratory analysis , not in paper 
def areaRegression():
    df=pd.read_csv('yelpAreaDataWithAllFeatures.csv',sep='\t')
    df=df[df.totalReviews>=20]
    Xvars=[u'categoryEntropy',
       u'averageCategoryDivergence', u'degreeCus',
       u'incomeEntropy',
       u'avgCusEntropy', u'pairwise', u'cusEigen']
    XvarsC=['totalReviews','degreeDist','numberofShops','numberofCustomers']
    Yvars=[u'income2010', u'income2011', u'income2012', u'income2013',
       u'income2014', u'income2015', u'incomePer2010', u'incomePer2011',
       u'incomePer2012', u'incomePer2013', u'incomePer2014', u'incomePer2015',
       u'pop2010', u'pop2011', u'pop2012', u'pop2013', u'pop2014', u'pop2015']
    for yvar in Yvars:
        df['log'+yvar]=np.log(df[yvar])
    df.drop(['GeoName'],1,inplace=True)
    df=df.apply(norm,0)
    for yvar in Yvars:
        regressionPlotsArea(yvar,Xvars,df)

    
if __name__ == "__main__":
    pass
#==============================================================================
# areaGraphConstruction()
#==============================================================================
#==============================================================================
# areaFeatureExtraction()
#==============================================================================
#==============================================================================
# areaRegression()
#==============================================================================

#%%

#==============================================================================
# 
# for y in [u'income2010', u'income2011', u'income2012', u'income2013',
#        u'income2014', u'income2015', u'incomePer2010', u'incomePer2011',
#        u'incomePer2012', u'incomePer2013', u'incomePer2014', u'incomePer2015',
#        u'pop2010', u'pop2011', u'pop2012', u'pop2013', u'pop2014', u'pop2015']:
#     os.system('mkdir {}'.format(y))
#==============================================================================
