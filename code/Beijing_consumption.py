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


def getCoupon2shop():
    data = pd.read_csv('meituan_data.csv', 'utf-8', delimiter = ',')
    coupon2shop = {}
    data.columns = range(16)
    for i in range(len(data)):
        try:
            coupon2shop[int(data[11][i].split('/')[-1][:-5])] = int(data[14][i].split('/')[-1])
        except:
            pass
    return coupon2shop

def getShop2idFreq(coupon2shop):
    shop2idf={}
    banlist=set()
    couponList = json.load(open('/home/yuan/bz_user.json', 'rb'))
    for key,value in couponList.iteritems():
        username=key.split('\t')[1]
        if value<=20:
            couponname=int(key.split('\t')[0])
            if couponname in coupon2shop:
                shopname=coupon2shop[couponname]
                if shopname not in shop2idf:
                    shop2idf[shopname]={}
                if username not in shop2idf[shopname]:
                    shop2idf[shopname][username]=0
                shop2idf[shopname][username]+=1
        else:
            banlist=banlist | set(username)
    banlist=list(banlist)    
    for key,value in shop2idf.iteritems():
        for user in banlist:
            if user in value:
                value.pop(user)        
    return shop2idf

def getShop2id(shop2idf):
    shop2id={}
    for key,dic in shop2idf.iteritems():
        lis=[]
        for user,value in dic.iteritems():
            lis.append(user)
        shop2id[key]=list(set(lis))
    return shop2id

def filterNodes(shop2id):
    meituan=pd.read_csv('../CleanData/combined_meituan_dianping_full.csv',encoding='utf-8')
    catCount=meituan['meituan_category'].value_counts()
    catCount=catCount[catCount>50]
    df=meituan['meituan_category'].isin(catCount.index.tolist())
    df=meituan[df]
    def inside(testlat,testlon):
        return 39.649111<=testlat<=40.513182 and 115.748255<=testlon<=117.440648
    def filtering(df):
        flags=df.apply(lambda row : inside(row['lat'],row['lon']),axis=1)
        df3=df[flags==True]
        return df3
    df3=filtering(df)
    dfID=df3['meituan_id'].tolist()
    allowedlist=[]
    newshop={}
    for key,value in shop2id.iteritems():
        if len(value)>=10 and key in dfID:
            allowedlist.append(key)
            newshop[key]=value
    dfID=df3['meituan_id'].isin(allowedlist)
    finalDf=df3[dfID]
    finalDf['popularity'] = finalDf[u'sales_revenue']/finalDf['dianping_price']
    shop={}
    shop2id_n={}
    for index,row in finalDf.iterrows():
        if row['meituan_id'] not in shop:
            idx=row['meituan_id']
            shop2id_n[idx]=shop2id[idx]
            shop[idx] = {}
            shop[idx]['num_sales'] = row[u'sales_volume']
            shop[idx]['category'] = row['meituan_category']
            shop[idx]['shopname'] = row['shopname']
            shop[idx]['rating']= row['rating']
            shop[idx]['sales_volume']=row['sales_volume']
            shop[idx]['sales_revenue'] = row[u'sales_revenue']
            shop[idx]['review_counts'] = row[u'review_counts']
            shop[idx]['dianping_price'] = row[u'dianping_price']
            shop[idx]['lat'] = row[u'lat']
            shop[idx]['lon'] = row[u'lon']
            shop[idx]['meituan_id'] = idx
            shop[idx]['popularity'] = row['popularity']
    return shop,shop2id_n

def getUser2shop(shop2id):
    user2shop = {}
    cnt=0
    user2idx={}
    for k in shop2id:
        for j in shop2id[k]:
            if j in user2shop:
                user2shop[j].append(k)
            else:
                user2shop[j] = [k]
                user2idx[j]=cnt
                cnt+=1
    return user2shop

def getIntersect(user2shop):
    tower_dict = {}
    now = datetime.now()
    all_values = user2shop.values()
    for u in range(len(user2shop)):
        if u%1000 == 0:
            print 'passed', u, 'users', datetime.now() - now
            now = datetime.now()
        if len(all_values[u]) == 1:
            continue
        tmp = sorted(all_values[u])
        for i in range(len(tmp)):
            for j in range(i + 1, len(tmp)):
                name = str(tmp[i]) + '_' + str(tmp[j])
                if name in tower_dict:
                    tower_dict[name] += 1
                else:
                    tower_dict[name] = 1
    return tower_dict

def getDistance():
    distDic=json.load(open('../CleanData/distanceDic.json', 'rb'))
    return distDic


## 3 functions for simulationpreprocessing
def getUserCentroid():
    return json.load(open('../CleanData/user_centroid.json', 'rb'))
                     
def computedistance(shopkey,centroid,shop):
    vals=(shop[shopkey]['lat'],shop[shopkey]['lon'])
    distance=vincenty(vals,(centroid[0],centroid[1])).km
    return distance
    
    
def getDataForParameterFitting(shop2id,user2shop,shop2idf,centroidDic):
    masterlist=shop2id.keys()
    NN=len(masterlist)
    dist2shop={}
    cnt=0
    toc=time.time()
    for key,value in user2shop.iteritems():
        if cnt % 1000 ==0:
            tic=time.time()
            print 'Done with 1000 in {} s'.format(tic-toc)
            toc=tic
        cnt+=1
        remaining=copy.deepcopy(masterlist)
        assert len(remaining)==NN
        N=len(value)
        dist2shop[key]={}
        dist2shop[key]['shopsPositive']=[]
        dist2shop[key]['distancesPositive']=[]
        dist2shop[key]['shopsNegative']=[]
        dist2shop[key]['distancesNegative']=[]
        for shop in value:
            dist2shop[key]['shopsPositive'].append(shop)
            dist2shop[key]['distancesPositive'].append(computedistance(shop,centroidDic[key]))
            remaining.remove(shop)
        selected=random.sample(remaining,N)
        for s in selected:
            dist2shop[key]['shopsNegative'].append(s)
            dist2shop[key]['distancesNegative'].append(computedistance(s,centroidDic[key]))
    idname=[]
    shopname=[]
    pij=[]
    dij=[]
    pop=[]
    freq=[]
    for key,value in dist2shop.iteritems():
        shopP=value['shopsPositive']
        distP=value['distancesPositive']
        p=[1 for i in range(len(shopP))]
        po=[shop[sh]['sales_volume'] for sh in shopP]
        fre=[shop2idf[sh][key] for sh in shopP]
        
        idname+=[key for i in range(len(shopP))]
        shopname+=shopP
        dij+=distP
        pop+=po
        pij+=p
        freq+=fre

        shopN=value['shopsNegative']
        distN=value['distancesNegative']
        p2=[0 for i in range(len(shopP))]
        po2=[shop[sh]['sales_volume'] for sh in shopN]
        fre2=[0 for sh in shopN]
        
        idname+=[key for i in range(len(shopN))]
        shopname+=shopN
        dij+=distN
        pop+=po2
        pij+=p2
        freq+=fre2
    df=pd.DataFrame(np.stack([idname,shopname,pop,dij,pij,freq],1),columns=['individual','shopname','popularity','distance','Pij','frequency'])
    return df
        
def getShopUserGridDistance(shop):
    mean=np.mean([39.649111,40.513182])
    latd= vincenty((39.649111,115.748255),(40.513182,115.748255)).km
    lond=vincenty((mean,115.748255),(mean,117.440648)).km
    latN=int(math.ceil(latd/0.5))
    lonN=int(math.ceil(lond/0.5))
    latspace=np.linspace(39.649111,40.513182,num=latN)
    lonspace=np.linspace(115.748255,117.440648,num=lonN)
    latGap=latd/(latN-1)
    lonGap=lond/(lonN-1)
    def assign(lat,lon):
        i=0
        j=0
        foundlat= latspace[i]<=lat<latspace[i+1] 
        foundlon=lonspace[j]<=lon<lonspace[j+1]
        while not foundlat:
            i+=1
            foundlat= latspace[i]<=lat<latspace[i+1]
        while not foundlon:
            j+=1
            foundlon= lonspace[j]<=lon<lonspace[j+1]
        return (i,j)
    filteredShopInfo=pd.DataFrame(shop.values())
    ids=filteredShopInfo['meituan_id'].tolist()
    lats=filteredShopInfo['lat'].tolist()
    lons=filteredShopInfo['lon'].tolist()
    vecAssign=np.vectorize(assign)
    assigned=vecAssign(np.array(lats),np.array(lons))
    shop2grid={}
    for i in range(len(ids)):
        shop2grid[ids[i]]=(assigned[0][i],assigned[1][i])
    df=json.load(open('../CleanData/user_centroid.json', 'rb'))
    ids=df.keys()
    lats=[float(i[0]) for i in df.values()]
    lons=[float(i[1]) for i in df.values()]
    def inside(testlat,testlon):
        return 39.649111<=testlat<=40.513182 and 115.748255<=testlon<=117.440648
    df=pd.DataFrame(np.stack([ids,lats,lons],1),columns=['id','lat','lon'])
    df['lat']=df['lat'].astype(float)
    df['lon']=df['lon'].astype(float)
    flags=df.apply(lambda row : inside(row['lat'],row['lon']),axis=1)
    df=df[flags==True]
    ids=df['id'].tolist()
    lats=df['lat'].tolist()
    lons=df['lon'].tolist()
    vecAssign=np.vectorize(assign)
    assigned=vecAssign(np.array(lats),np.array(lons))
    user2grid={}
    for i in range(len(ids)):
        user2grid[ids[i]]=(assigned[0][i],assigned[1][i])
    def getGridDistance(one,two,latGap,lonGap):       
        x=(one[0]-two[0])*latGap
        y=(one[1]-two[1])*lonGap
        distance=math.sqrt(x*x+y*y)
        if distance==0:
            distance=math.sqrt(latGap*lonGap)
        return distance
    nonempty=pd.Series(user2grid.values()).unique()       
    shopuserDist={}
    for key,value in shop2grid.iteritems():
        shopgrid=value
        for usergrid in nonempty:
            shopuserDist[(key,usergrid)]=getGridDistance(shopgrid,usergrid,latGap,lonGap)
    return shopuserDist

def preprocessing(sim=False):
    coupon2shop=getCoupon2shop()
    shop2idf=getShop2idFreq(coupon2shop)
    shop2id=getShop2id(shop2idf)
    shop,shop2id,=filterNodes(shop2id)
    user2shop=getUser2shop(shop2id)
#==============================================================================
#     intersect_all=getIntersect(user2shop)
#     distDic=getDistance()
#     json.dump(intersect_all, open('intersect_all.json', 'wb'))
#     json.dump(distDic, open('distDic.json', 'wb'))
#==============================================================================
    json.dump(shop, open('../CleanData/shop.json', 'wb'))
    json.dump(shop2id, open('../CleanData/shop2id.json', 'wb'))
    json.dump(user2shop, open('../CleanData/user2shop.json', 'wb'))
    if sim==True:
        centroidDic=getUserCentroid()
        df=getDataForParameterFitting(shop2id,user2shop,shop2idf,centroidDic)
        df.to_csv('AlphaBetaFitting.csv',encoding='utf-8',sep='\t',index=False)
        shopUserDistDic=getShopUserGridDistance(shop)
        json.dump(shopUserDistDic, open('../CleanData/shopUserDistDic.json', 'wb'))



def getarea2id(shop,shop2id):
    df2=pd.read_csv('../CleanData/meituan_district_mapping.csv',encoding='utf-8')
    name2num={}
    num2name={}
    shopID2areaID={}
    for index,row in df2.iterrows():
        name=row['districtName']
        ID=row['districtID']
        shopID2areaID[row['meituan_id']]=ID
        if name not in name2num:
            name2num[name]=ID
            num2name[ID]=name
    df=pd.DataFrame(shop.values())
    area2id={}
    for index,row in df.iterrows():
        s=row['meituan_id']
        area= shopID2areaID[s]
        if area not in area2id:
            area2id[area]={}
        for idx in shop2id[s]:
            if idx not in  area2id[area]:
                area2id[area][idx]=0
            area2id[area][idx]+=1
    for key in list(shopID2areaID.keys()):
        if shopID2areaID[key]==0:
            shopID2areaID.pop(key)
    for key in list(name2num.keys()):
        if name2num[key]==0:
            name2num.pop(key)
    areaID2shopID={}
    for shopID,areaID in shopID2areaID.iteritems():
        if areaID not in areaID2shopID:
            areaID2shopID[areaID]=[]
        areaID2shopID[areaID].append(shopID)    
    area2id.pop(0)
    num2name.pop(0)    
    return area2id,name2num,num2name,shopID2areaID,areaID2shopID

def getAreaGPS(shop,areaID2shopID):
    df= pd.DataFrame(shop.values())
    area2location = {}
    shop2location = {}
    for index,row in df.iterrows():
        x = row['lat']
        y = row['lon']
        shop2location[row['meituan_id']] = (float(x), float(y))
    for area in areaID2shopID:
        for business in areaID2shopID[area]:
            if shop2location[business] not in area2location:
                area2location[area] = []
            area2location[area].append(shop2location[business])
    return area2location,shop2location

def buildAreaGraph(graph_type,area2id,area2location,num2name):
    if graph_type=='customer':
        G = nx.Graph()
        Is=[]
        Js=[]
        Inters=[]
        Jacs=[]
        for area in area2id:
            G.add_node(area)
        for a1 in area2id:
            for a2 in area2id:
                if a1 < a2:
                    s1 = set(area2id[a1].keys())
                    s2 = set(area2id[a2].keys())
                    if len(s1) > 0 and len(s2) > 0:
                        inter = len(s1 & s2)
                        union = len(s1) + len(s2) - inter
                        if inter > 0:
                            G.add_edge(a1, a2, weight=1.0*inter/union)
                            Is.append(a1)
                            Js.append(a2)
                            Inters.append(inter)
                            Jacs.append(1.0*inter/union)
        df=pd.DataFrame(np.stack([Is,Js,Inters,Jacs],1),columns=['I','J','weight_raw','weight_jaccard'])
        df.to_csv('../CleanData/area_graph_edge.csv',sep='\t',index=False)
        return G
    elif graph_type == 'dist':
        G_dist=nx.Graph()
        thres=10
        area2central = {}
        for node in area2location:
            x = np.mean([xx[0] for xx in area2location[node]])
            y = np.mean([xx[1] for xx in area2location[node]])
            area2central[node] = (x,y)
            G_dist.add_node(node)
        for u in num2name:
            for v in num2name:
                if u < v:
                    x1 = area2central[u][0]
                    y1 = area2central[u][1]
                    x2 = area2central[v][0]
                    y2 = area2central[v][1]
                    distance = vincenty((x1, y1), (x2, y2)).kilometers
                    if distance<thres:
                        G_dist.add_edge(u,v,weight=1/distance)
        return G_dist

def areaGraphConstruction():
    shop_temp=json.load(open('shop.json','rb'))
    shop2id_temp=json.load(open('shop2id.json','rb'))
    shop,shop2id={},{}
    for key,value in shop_temp.iteritems():
        shop[int(key)]=value
    for key,value in shop2id_temp.iteritems():
        shop2id[int(key)]=value
    area2id,name2num,num2name,shopID2areaID,areaID2shopID=getarea2id(shop,shop2id)
    area2location,id2location=getAreaGPS(shop,areaID2shopID)
    G_cus_area=buildAreaGraph('customer',area2id,area2location,num2name)
    G_dist_area=buildAreaGraph('dist',area2id,area2location,num2name)
    
    json.dump(area2id, open('../CleanData/area2id.json', 'wb'))
    json.dump(areaID2shopID, open('../CleanData/areaID2shopID.json', 'wb'))
    json.dump(shopID2areaID, open('../CleanData/shopID2areaID.json', 'wb'))
    nx.write_gpickle(G_cus_area,'../CleanData/G_cus_area.pkl')
    nx.write_gpickle(G_dist_area,'../CleanData/G_dist_area.pkl')
#%%

def getArea2category(areaID2shopID,shop):
    df = pd.DataFrame(shop.values())
    shop2category = {}
    for index,row in df.iterrows():
        mid=row['meituan_id']
        shop2category[mid] = row['category']
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

def getCustomerIncomeEntropy(area2id,user2shop,shop):
    incomeData=pd.DataFrame(shop.values())
    incomeData=incomeData[incomeData['dianping_price'].notnull()]
    flag=incomeData['category'].isin([u'甜点饮品'])
    incomeData=incomeData[~flag]
    priceDict={}
    for index, row in incomeData.iterrows():
        priceDict[row['meituan_id']]=row['dianping_price']
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
    category=pd.qcut(userIncome,4,labels=False).tolist()
    incomeDic={}
    for i in range(len(category)):
        incomeDic[userID[i]]=userIncome[i]
    for i in emptyID:
        incomeDic[i]='NoInfo'
    IncomeEntropyID=[]
    IncomeEntropy=[]
    Num_customers=[]
    for area,iddict in area2id.iteritems():
        count_dict = {}
        for name,times in iddict.iteritems():
            if incomeDic[name] not in count_dict:
                count_dict[incomeDic[name]] = times
            else:
                count_dict[incomeDic[name]] += times
        cnt = [1.0 * i/sum(count_dict.values()) for i in count_dict.values()]
        entropy = 0#sc.stats.entropy(cnt)
        for e in cnt: 
            entropy -= e * np.log(e)
        IncomeEntropy.append(entropy)
        IncomeEntropyID.append(area)
        Num_customers.append(np.sum(iddict.values()))
    area2income={}
    area2numCus={}
    for i in range(len(IncomeEntropyID)):
        area2income[IncomeEntropyID[i]]=IncomeEntropy[i]
        area2numCus[IncomeEntropyID[i]]=Num_customers[i]
    return area2income,area2numCus

def getDessertEntropy(area2category):
    area2ratio={}
    area2biCat={}
    for area,catDic in area2category.iteritems():
        total=np.sum(catDic.values())
        try:
            dessert=catDic[u'甜点饮品']
        except:
            dessert=0
        area2ratio[area]=dessert*1.0 /total
        area2biCat[area]={'dessert': dessert*1.0/total, 'others': (total-dessert)*1.0/total}
    return area2ratio,area2biCat

def getAreaCustomerEntropy(user2shop,shop,area2id):
    cusEntro={}
    for key,value in user2shop.iteritems():
        entro={}
        for v in value:
            shopvcat=shop[v]['category']
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
    for area,iddict in area2id.iteritems():
        areaCusEntropy[area]=np.mean([count*cusEntro[name] for name,count in iddict.iteritems()])
    return areaCusEntropy

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

def getIndptVars(shop,shopID2areaID):
    df=pd.DataFrame(shop.values())
    df2=pd.read_csv('../CleanData/meituan2district_withInfo.csv',encoding='utf-8')
    area2pop={}
    area2sale={}
    area2rate={}
    area2econ={}
    area2vol={}
    for index,row in df2.iterrows():
        area=row['districtID']
        if area not in area2econ:
            area2econ[area]=row['economy']
    area2econ.pop(0)
    area=[]
    for index,row in df.iterrows():
        try:
            area.append(shopID2areaID[row['meituan_id']])
        except:
            area.append(0)
    df=pd.concat([df,pd.DataFrame(np.stack([area],1),columns=['area'])],1)  
    grp=df.groupby(['area'])
    for name,group in grp:
        index=name
        area2pop[index]=group['popularity'].sum()
        area2sale[index]=group[u'sales_revenue'].sum()
        area2rate[index]=group[u'rating'].mean()
        area2vol[index]=group['sales_volume'].sum()
    area2pop.pop(0)
    area2sale.pop(0)
    area2rate.pop(0)
    return area2pop,area2sale,area2rate,area2econ,area2vol

def areaAddGraphFeatures(area2entropy,G_cus_area,G_dist_area,area2econ,area2sales,area2pop,area2rating,area2numCus,area2income,area2ratio,areaCusEntropy,area2biCat,areaID2shopID,area2Pairwise,area2category,area2vol):
    econ=[]
    sales=[]
    pop=[]
    ratings=[]
    entro=[]
    diverge=[]
    degreeCus=[]
    degreeDist=[]
    numCus=[]
    incomeEntro=[]
    avgCusEntro=[]
    dessertRatio=[] 
    sameCat=[]
    numShops=[]
    Pairwise=[]
    vols=[]
    ids=area2entropy.keys()    
    for key in area2entropy:
        if len(G_cus_area.neighbors(key)) == 0:
           print 'weird that it happened'
           vols.append(area2vol[key])
           econ.append(area2econ[key])
           sales.append(area2sales[key])
           pop.append(area2pop[key])
           ratings.append(area2rating[key])
           entro.append(area2entropy[key])
           diverge.append(None)
           degreeCus.append(None)
           degreeDist.append(None)
           numCus.append(area2numCus[key])
           incomeEntro.append(area2income[key])
           avgCusEntro.append(None)
           dessertRatio.append(area2ratio[key])
           sameCat.append(None)
           numShops.append(0)
           Pairwise.append(0)
        else:
            vols.append(area2vol[key])
            econ.append(area2econ[key])
            sales.append(area2sales[key])
            pop.append(area2pop[key])
            ratings.append(area2rating[key])
            entro.append(area2entropy[key])
            diverge.append(np.mean([getDivergence(key,nei,area2category) for nei in  G_cus_area.neighbors(key)]))
            degreeCus.append(sum([G_cus_area[key][i]['weight'] for i in G_cus_area.neighbors(key)]))
            degreeDist.append(sum([G_dist_area[key][i]['weight'] for i in G_dist_area.neighbors(key)]))
            numCus.append(area2numCus[key])
            incomeEntro.append(area2income[key])
            avgCusEntro.append(areaCusEntropy[key])
            dessertRatio.append(area2ratio[key])
            sameCat.append(np.mean([(area2biCat[i]['dessert']-area2biCat[key]['dessert'])**2 + (area2biCat[i]['others']-area2biCat[key]['others'])**2 for i in G_cus_area.neighbors(key)]))
            numShops.append(len(areaID2shopID[key]))
            Pairwise.append(area2Pairwise[key])
    finalDf=pd.DataFrame(np.stack([ids,vols,econ,sales,pop,ratings,entro,diverge,degreeCus,degreeDist,numCus,numShops,incomeEntro,avgCusEntro,dessertRatio,sameCat,Pairwise],1),
                       columns=['districtID','totalVolume','economy','totalSales','totalPopularity','averageRating','categoryEntropy','averageCategoryDivergence','degreeCus','degreeDist','numberofCustomers','numberofShops','incomeEntropy',
                                'avgCusEntropy','fracofDessert','dessertCS','pairwise'])
    print 'computing eigen for cus net'
    centrality_cus = nx.eigenvector_centrality(G_cus_area,max_iter=100,tol=0.001)
    print 'computing eigen for dist net'
    centrality_dist = nx.eigenvector_centrality(G_dist_area,max_iter=100,tol=0.001)
    cusKey=centrality_cus.keys()
    cusVal=centrality_cus.values()
    dKey=centrality_dist.keys()
    dVal=centrality_dist.values()
    df1=pd.DataFrame(np.stack([cusKey,cusVal],1),columns=['districtID','cusEigen'])
    df2=pd.DataFrame(np.stack([dKey,dVal],1),columns=['districtID','dEigen'])
    addDf=pd.merge(df1,df2,how='left',on='districtID')
    finalDf=pd.merge(finalDf,addDf,how='left',on='districtID')    
    return finalDf

def areaFeatureExtraction():
    shopID2areaID_temp=json.load(open('../CleanData/shopID2areaID.json','rb'))
    shopID2areaID={}
    for key,value in shopID2areaID_temp.iteritems():
        shopID2areaID[int(key)]=value
    shop_temp=json.load(open('../CleanData/shop.json','rb'))
    shop={}
    for key,value in shop_temp.iteritems():
        shop[int(key)]=value
    areaID2shopID_temp=json.load(open('../CleanData/areaID2shopID.json','rb'))
    areaID2shopID={}
    for key,value in areaID2shopID_temp.iteritems():
        areaID2shopID[int(key)]=value
    area2id_temp=json.load(open('../CleanData/area2id.json','rb'))
    area2id={}
    for key,value in area2id_temp.iteritems():
        area2id[int(key)]=value
    user2shop=json.load(open('../CleanData/user2shop.json','rb'))
    area2category,area2entropy=getArea2category(areaID2shopID,shop)
    area2income,area2numCus=getCustomerIncomeEntropy(area2id,user2shop,shop)
    area2ratio,area2biCat=getDessertEntropy(area2category)
    areaCusEntropy=getAreaCustomerEntropy(user2shop,shop,area2id)
    G_cus_area=nx.read_gpickle('../CleanData/G_cus_area.pkl')
    G_dist_area=nx.read_gpickle('../CleanData/G_dist_area.pkl')
    area2pairwise=getPairWiseDiff(area2category,G_cus_area)
    area2pop,area2sales,area2rating,area2econ,area2vol=getIndptVars(shop,shopID2areaID)
    df=areaAddGraphFeatures(area2entropy,G_cus_area,G_dist_area,area2econ,area2sales,area2pop,area2rating,area2numCus,area2income,area2ratio,areaCusEntropy,area2biCat,areaID2shopID,area2pairwise,area2category,area2vol)
    df.to_csv('../CleanData/beijing_consumption.csv',encoding='utf-8',sep='\t',index=False)
    
#%%
def normaliseData(df,fet_list):
    df2=copy.deepcopy(df)
    df2=df2.drop(fet_list,1)
    newcols=[list(df2.districtID)]
    for colname in fet_list:
        series=df[colname]
        series=series-(series.mean())
        std=np.std(series)
        series=series/std
        newcols.append(list(series))    
    newData=pd.DataFrame(np.stack(newcols,1),columns=['districtID']+fet_list)
    newData=pd.merge(df2,newData,how='left',on='districtID')
    return newData

def regressionPlotsArea(yvar,fet_list,df,df_norm,norm=False):
    if norm:
        feature = df_norm
        featureS= feature[feature.totalSales >0][['totalSales','districtID']]
        featureS['log_sales']=np.log(featureS.totalSales)
        featureS['log_sales']=featureS['log_sales']-featureS['log_sales'].mean()
        featureS['log_sales']=featureS['log_sales']/np.std(list(featureS['log_sales']))
        featureS=featureS.drop('totalSales',1)
        feature=pd.merge(feature,featureS,on='districtID',how='left')
        featureS= feature[feature.totalPopularity>0][['totalPopularity','districtID']]
        featureS['log_pop']=np.log(featureS.totalPopularity)
        featureS['log_pop']=featureS['log_pop']-featureS['log_pop'].mean()
        featureS['log_pop']=featureS['log_pop']/np.std(list(featureS['log_pop']))
        featureS=featureS.drop('totalPopularity',1)
        feature=pd.merge(feature,featureS,on='districtID',how='left') 
        featureS= feature[feature.economy>0][['economy','districtID']]
        featureS['log_econ']=np.log(featureS.economy)
        featureS['log_econ']=featureS['log_econ']-featureS['log_econ'].mean()
        featureS['log_econ']=featureS['log_econ']/np.std(list(featureS['log_econ']))
        featureS=featureS.drop('economy',1)
        feature=pd.merge(feature,featureS,on='districtID',how='left')
        featureS= feature[feature.totalVolume>0][['totalVolume','districtID']]
        featureS['log_vol']=np.log(featureS.totalVolume)
        featureS['log_vol']=featureS['log_vol']-featureS['log_vol'].mean()
        featureS['log_vol']=featureS['log_vol']/np.std(list(featureS['log_vol']))
        featureS=featureS.drop('totalVolume',1)
        feature=pd.merge(feature,featureS,on='districtID',how='left')
        feature['averageRating']=feature['averageRating']-feature['averageRating'].mean()
        feature['averageRating']=feature['averageRating']/np.std(list(feature['averageRating']))
    else:
        feature = df
        featureS= feature[feature.totalSales>0][['totalSales','districtID']]
        featureS['log_sales']=np.log(featureS.totalSales)
        featureS=featureS.drop('totalSales',1)
        feature=pd.merge(feature,featureS,on='districtID',how='left')
        featureS= feature[feature.totalPopularity>0][['totalPopularity','districtID']]
        featureS['log_pop']=np.log(featureS.totalPopularity)
        featureS=featureS.drop('totalPopularity',1)
        feature=pd.merge(feature,featureS,on='districtID',how='left')
        featureS= feature[feature.economy>0][['economy','districtID']]
        featureS['log_econ']=np.log(featureS.economy)
        featureS=featureS.drop('economy',1)
        feature=pd.merge(feature,featureS,on='districtID',how='left')
        featureS= feature[feature.totalVolume>0][['totalVolume','districtID']]
        featureS['log_vol']=np.log(featureS.totalVolume)
        featureS=featureS.drop('totalVolume',1)
        feature=pd.merge(feature,featureS,on='districtID',how='left')
    for i in range(len(fet_list)):
        fet_int = fet_list[i]
        plt.figure(num=None, figsize=(10, 5), dpi=80, edgecolor='k')
        feature_sub = feature 
        plt.subplot(121)
        plt.scatter(feature_sub[fet_int].values, feature_sub[yvar].values, alpha = 0.3)
        plt.ylabel(yvar)
        plt.xlabel('{}'.format(fet_int))
        plt.title('{} vs. {}'.format(yvar,fet_int))        
        model = ols("{} ~ degreeDist + numberofCustomers +numberofShops ".format(yvar), feature).fit()
        feature['{}_resid'.format(yvar)] = model.resid
        feature['{}_resid'.format(yvar)] = feature['{}_resid'.format(yvar)] - feature['{}_resid'.format(yvar)].mean()
        feature['{}_resid'.format(yvar)] = feature['{}_resid'.format(yvar)]/ np.std(feature['{}_resid'.format(yvar)])
        model = ols("{}_resid ~  {}".format(yvar,fet_int), feature_sub).fit()
        plt.subplot(122)
        plt.scatter(feature_sub[fet_int].values, feature_sub['{}_resid'.format(yvar)].values, alpha = 0.3)
        plt.ylabel('residuals after controlling')
        plt.xlabel('{}'.format(fet_int))
        plt.title('{} vs. {}'.format(yvar,fet_int))
        plt.savefig('../plots/area/{}/{}_{}.pdf'.format(yvar,yvar,fet_int))
        print feature_sub[[fet_int, '{}_resid'.format(yvar)]].corr()
        print(model.summary())
        
def areaRegression():
    df = pd.read_csv('beijing_consumption.csv',encoding='utf-8',sep='\t')
    fet_list = [u'categoryEntropy',u'averageCategoryDivergence', u'degreeCus', 
         u'degreeDist',u'incomeEntropy', u'avgCusEntropy', u'dessertCS', u'cusEigen', u'dEigen','fracofDessert','pairwise']
    yvars=['log_pop','log_econ','averageRating','log_sales']
    df_norm=normaliseData(df,fet_list)
    for yvar in yvars:
        regressionPlotsArea(yvar,fet_list,df,df_norm,norm=True)


def addAverageIncome():
    user2shop=json.load(open('../UserAnalysis/user2shop.json', 'rb'))
    area2id=json.load(open('area2id.json', 'rb'))
    
    incomeData=pd.read_csv('../CleanData/datawithAllfeatures.csv',encoding='utf-8',sep='\t')
    incomeData=incomeData[incomeData['dianping_price'].notnull()]
    flag=incomeData['meituan_category'].isin([u'甜点饮品'])
    incomeData=incomeData[~flag]
    priceDict={}
    for index, row in incomeData.iterrows():
        priceDict[row['meituan_id']]=row['dianping_price']
    userIncome=[]
    userAvgIncome=[]
    userID=[]
    for key,value in user2shop.iteritems():
        summ=0.0
        count=0
        for shop in list(value):
            if shop in priceDict:
                summ+=priceDict[shop]
                count+=1
        if count!=0:
            userIncome.append(summ)
            userID.append(key)
            userAvgIncome.append(summ/count)

    incomeDic={}
    avgincomeDic={}
    for i in range(len(userIncome)):
        incomeDic[userID[i]]=userIncome[i]
        avgincomeDic[userID[i]]=userAvgIncome[i]
        
    AreaID=[]
    averageTotalExpenditure=[]
    averageExpenditure=[]
    
    for area,iddict in area2id.iteritems():
        count_dict = {}
        count_dict_a = {}
        for name,times in iddict.iteritems():
            if name in incomeDic:
                if incomeDic[name] not in count_dict:
                    count_dict[incomeDic[name]] = times
                else:
                    count_dict[incomeDic[name]] += times
                if avgincomeDic[name] not in count_dict_a:
                    count_dict_a[avgincomeDic[name]] = times
                else:
                    count_dict_a[avgincomeDic[name]] += times
        ainc=0.0
        att=0
        inc=0.0
        tt=0
        for key,value in count_dict.iteritems():
            inc+=key*value
            tt+=value
            
        for key,value in count_dict_a.iteritems():
            ainc+=key*value
            att+=value
        averageTotalExpenditure.append(inc/tt)
        AreaID.append(area)
        averageExpenditure.append(ainc/att)
        
    df=pd.DataFrame(np.stack([AreaID,averageExpenditure,averageTotalExpenditure],1),columns=['district_id','avgExpenditure','averageTotalExpenditure'])
    df.to_csv('beijingEstIncome.csv',index=False)
    
    


def getPopulation():
    df=pd.read_csv('bj_pop.csv')[['districtID','Popu','Popu_float','Pop_Sum','Dens_Pob']]
    df.rename(columns={'districtID':'district_id','Pop_Sum':'pop','Popu_float':'popfloat','Popu':'popres'},inplace=True)
    df2=pd.read_csv('td/bj_house_price.csv')
    df2.rename(columns={'districtID':'district_id',u'Mean House Price (MHP)':'HPI'},inplace=True)
    df=pd.merge(df,df2[['district_id','HPI']],on='district_id',how='left')
    df['pop']=df['pop']/1000
    return df

def generateConsumptionDataWithControls():
    df=pd.read_csv('beijing_consumption.csv',sep='\t')
    df.rename(columns={'districtID':'district_id'},inplace=True)
    df.dropna(subset=['economy','categoryEntropy'],inplace=True)
    df['logEcon']=np.log(df.economy)

    df2=pd.read_csv('beijingEstIncome.csv')
    df=pd.merge(df,df2,on='district_id',how='left')
    
    df2=getPopulation()
    df=pd.merge(df,df2,on='district_id',how='left')
    
    df2=getEcon()[['district_id','econdse','area']]
    df=pd.merge(df,df2,on='district_id',how='left')
    df['logEcondse']=np.log(df.econdse)
    
    df['popdse']=df['pop']/df['area']
    df['logpopdse']=np.log(df['popdse'])
    
    rents=df[df.HPI!=0].HPI.min()
    df.HPI=df.HPI/rents
    
#===============================================co===============================
#     df=df[df['numberofShops']!=1]
#==============================================================================
#==============================================================================
#     df=df[df['categoryEntropy']!=0]
#==============================================================================
    return df
    
#%%
areaGraphConstruction()
areaFeatureExtraction()
addAverageIncome()
generateConsumptionDataWithControls()
#==============================================================================
# areaRegression()
#==============================================================================
