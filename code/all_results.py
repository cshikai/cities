#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:05:48 2018

@author: kai
"""
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import networkx as nx
import json

import folium

import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

sns.set(color_codes=True)
sns.set_style("whitegrid")

EXCLUDE = [1,131,111]

def split_df():
    rows=1000000
    #split file because of github limit
    df= pd.read_csv('../data/istanbul/0-transactions.txt')
    for i in range(int(np.ceil(len(df)/rows))):
         df.iloc[i*rows:(i+1)*rows].to_csv('../data/istanbul/0-transactions_{}.txt'.format(i),index=False)
         

def normalise_data(df,omit=[]): 
    df = df.copy(deep=True)
    cols=df.columns.tolist()
    for o in omit:
        cols.remove(o)
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col]-mean)/std
    return df

#helper to read raw credit card data , repeat means we consider more than one trip by a user
def read_raw_flows(is_repeat,omit=True):
    print("Reading transcation data...\n")
    location = '../data/istanbul/'
    df_trans = pd.DataFrame()
    for f in os.listdir(location):
        if '0-transactions_' in f:
            df_trans= df_trans.append(pd.read_csv(os.path.join(location,f)))
    
    df_trans.rename(columns={
        'MUSTERI_ID_MASK': 'customer_id',
        'UYEISYERI_ID_MASK': 'shop_id',
        'ISLEM_TARIHI': 'date',
        'ISLEM_SAAT': 'time',
        'ONLINE_ISLEM': 'is_online',
        'HARCAMA_TIPI': 'spending_type',
        'ISLEM_TUTARI': 'transcation_amount',
        },inplace=True)
    
    len1 = len(df_trans)
    df_trans = df_trans[(df_trans['shop_id']!=999999) & (df_trans.is_online ==0)]
    
    
    df_shop = pd.read_csv('../data/istanbul/5-shopiddistid.csv')
   
    df_trans = pd.merge(df_trans,df_shop,on='shop_id',how='left')
    df_cust = pd.read_csv('../data/istanbul/6-custidhdistid.csv')
    df_trans = pd.merge(df_trans,df_cust,on='customer_id',how='left')
    if not is_repeat:
        df_trans.drop_duplicates(subset=['shop_id','customer_id'],inplace=True)
    
    df_trans.dropna(subset=['hdistrict_id','sdistrict_id'],inplace=True)
    
    
    print('After filtering out shops with no ids and online transactions, there is {}percent left\n'.format(len(df_trans)/len1*100))
       
    print('Reading and filtering complete!')
    print('There are {} customers in the dataset'.format(df_trans.customer_id.nunique()))
    print('There are {} shops in the dataset'.format(df_trans.shop_id.nunique()))
    print('There are {} entries dataset'.format(len(df_trans)))
    df_trans['sdistrict_id'] = df_trans.sdistrict_id.astype(int)
    df_trans['hdistrict_id'] = df_trans.hdistrict_id.astype(int)
    
    if omit:
        df_trans = df_trans[~(df_trans.sdistrict_id.isin(EXCLUDE)|df_trans.hdistrict_id.isin(EXCLUDE))]
    return df_trans

def population_representation(omit=True):
    df = read_raw_flows(False,omit)
    population_rep = df.groupby('hdistrict_id')['customer_id'].nunique().rename('pop_rep')
    population_rep.index.rename('district_id',inplace=True)
    return population_rep

def get_work_flow():
    df_home = pd.read_csv('../data/istanbul/6-custidhdistid.csv')
    df_work = pd.read_csv('../data/istanbul/7-custidwdistid.csv')
    df=pd.merge(df_home,df_work,how='left',on='customer_id')
    movement = df.groupby(['hdistrict_id','wdistrict_id']).apply(len).rename('workflow').reset_index()
    exclude = [1,131,111]
    
    movement = movement[~(movement.wdistrict_id.isin(exclude)|movement.hdistrict_id.isin(exclude))]
    return movement

# helper to get customer information
def get_customer_info():
    print ('Reading customer demographic data...\n')
    #62392 unique customers
    demog = pd.read_csv('../data/istanbul/8-customers_demographics.csv')
    demog['agecat'] = demog.age//10
    #fill in missing income information with average income
    mask = demog.income == 0
    avgincome = demog.income[demog.income!=0].mean()
    demog.loc[mask,'income'] = avgincome
    #income decile
    demog['incomecat']=pd.qcut(demog.income,10,labels=False)
    
    df_home = pd.read_csv('../data/istanbul/6-custidhdistid.csv')
    demog = pd.merge(demog,df_home,how='left',on='customer_id')
    df_work = pd.read_csv('../data/istanbul/7-custidwdistid.csv')
    demog=pd.merge(demog,df_work,how='left',on='customer_id')
    demog.dropna(inplace=True)
    assert demog.customer_id.is_unique
    return demog

def get_district_level_income():
    df = pd.read_csv('../data/istanbul/8-customers_demographics.csv')
    df = df[df.income!=0]
    df_home = pd.read_csv('../data/istanbul/6-custidhdistid.csv')
    df = pd.merge(df,df_home,on='customer_id',how='left')
    df_work = pd.read_csv('../data/istanbul/7-custidwdistid.csv')
    df = pd.merge(df,df_work,how='left',on='customer_id')
    g=df.groupby('hdistrict_id')['income'].mean().rename('hdistrict_income').reset_index()
    g.rename(columns = {'hdistrict_id':'district_id'},inplace=True)
# =============================================================================
#     g.index.rename('district_id',inplace=True)
# =============================================================================
    g2=df.groupby('wdistrict_id')['income'].mean().rename('wdistrict_income').reset_index()
    g2.rename(columns = {'wdistrict_id':'district_id'},inplace=True)
# =============================================================================
#     g2.index.rename('district_id',inpla)
# =============================================================================

    #g.to_csv('istanbulData/districtIncomeData.csv',index=False)
    return pd.merge(g2,g,on='district_id',how='left')



# helper to get flow matrix. hdistrict_id = origin. sdistrict_id = destination
def get_flow_matrix(is_repeat):
    df_trans = read_raw_flows(is_repeat)
    movement = df_trans.groupby(['hdistrict_id','sdistrict_id']).apply(len).rename('flow').reset_index()

    return movement

def get_flow_df(is_repeat):
    flowmat = get_total_flows(is_repeat=True)
    outflow = flowmat.groupby('district_i')['totalflow'].sum().rename('outflow')
    outflow.index.rename('district_id',inplace=True)
    
    inflow = flowmat.groupby('district_j')['totalflow'].sum().rename('inflow')
    inflow.index.rename('district_id',inplace=True)
    
    within_flow = flowmat[flowmat.district_i == flowmat.district_j]
    within_flow.rename(columns= {'totalflow': 'withinflow','district_i':'district_id'},inplace=True)
    within_flow.index=within_flow.district_id
    
    
    within_flow = within_flow[['withinflow']]
    
    df = within_flow.join(outflow,how ='left',on = 'district_id')
    df = df.join(inflow,how ='left',on = 'district_id')
    df.loc[:,'inflow'] = df['inflow'] - df['withinflow']
    df.loc[:,'outflow'] = df['outflow'] - df['withinflow']
    return df

def get_all_district_ids():
    df_home = pd.read_csv('../data/istanbul/6-custidhdistid.csv').hdistrict_id.rename('id')
    df_work = pd.read_csv('../data/istanbul/7-custidwdistid.csv').wdistrict_id.rename('id')
    df_shop = pd.read_csv('../data/istanbul/5-shopiddistid.csv').sdistrict_id.rename('id')
    df = pd.DataFrame((df_home.append(df_shop)).append(df_work).unique(),columns = ['district_id']).sort_values('district_id')
    df=df.reset_index(drop=True)
    
    df = df[~df['district_id'].isin(EXCLUDE)]
    return df
    
def diversity(chunk):
    return scipy.stats.entropy(chunk.counts.values)
    
def num_unique_diversity(chunk):
    return len(chunk) 

def total(chunk):
    return chunk.counts.sum()

def compute_diversity(df,district_col,diversity_col,diversity_type,column_name):
    if diversity_type == 'entropy':
        f = diversity
    elif diversity_type == 'num_unique':
        f = num_unique_diversity
    elif diversity_type == 'total':
        f = total
    groups=df.groupby([district_col,diversity_col]).apply(len).rename('counts').reset_index()    
    entros_in= groups.groupby([district_col]).apply(f).rename(column_name)
    entros_in.index.rename('district_id',inplace=True)
    return entros_in
    
def compute_all_diversity(df,district_col,diversity_col,column_name_prefix):
    
    df_div = get_all_district_ids()
    df_div.index =df_div.district_id
    df_div.drop(columns = ['district_id'],inplace=True)
    diversity_types = ['entropy','num_unique', 'total']
    for d in diversity_types:
        df_div=df_div.join(compute_diversity(df,district_col,diversity_col,d,'{}_{}'.format(column_name_prefix,d)),how='left',on='district_id')
    
    return df_div



        
        

# helper to get inflow demographic diversity,  and outflow destination diversity
def read_in_out_diversity(is_repeat=True):
    demog = get_customer_info()
    demog.index = demog.customer_id
    demog.drop(columns=['customer_id','hdistrict_id'],inplace=True)
    df = read_raw_flows(is_repeat)
    df = df.join(demog,on='customer_id',how='left') 
    assert not df.incomecat.isnull().sum()
    #df.dropna(subset=['hdistrict_id'],inplace=True)
    #'vector' that describes the diversity of the people
    df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) 
# =============================================================================
#     df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) + df.gender.astype(str)
# =============================================================================
# =============================================================================
#     df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.wdistrict_id.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) + df.gender.astype(str)
# =============================================================================
# =============================================================================
#     df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) + df.gender.astype(str)
# =============================================================================
    df_div=compute_all_diversity(df,'sdistrict_id','composite_in','composite_demographic_inflow')
    df_div=df_div.join(compute_all_diversity(df,'hdistrict_id','sdistrict_id','destination'),how='left',on='district_id')
    df_div.to_csv('../data/istanbul/flow_diversity.csv')
    return df_div

# diversity of avaialblity of shops - ie not weighted by consumption value 
def read_availablity_diversity():
    df = read_raw_flows(is_repeat=True)
    df2 = pd.read_csv("../data/istanbul/4-shopid_mccmerged.csv")
    df=pd.merge(df,df2,how="left",on="shop_id")
    df.drop_duplicates(subset = ['shop_id'],inplace=True)
    #merged categories that is more general
    df_div=compute_all_diversity(df,'sdistrict_id','mcc_merged','mcc_merged')
    df_div=df_div.join(compute_all_diversity(df,'sdistrict_id','mcc_detailed','mcc_detailed'),on='district_id',how='left')
    df_div.to_csv('../data/istanbul/shopavaildiversity.csv')
    return df_div

#build network based on flow matrix. returns        
def buildSimpleNetwork(df):
    def mapp(dic,name):
        for key ,value in dic.iteritems():
            ddf[key][name]=value
                
    ddf={}
    
    G=nx.DiGraph()
    GI=nx.DiGraph()
    nodes=df.hdistrict_id.unique()
    for n in nodes:
        ddf[n]={}
    names=[]
    dics=[]
        
    G.add_nodes_from(nodes)
    for index,row in df.iterrows():
        G.add_edge(row.hdistrict_id,row.sdistrict_id,{'weight':row.counts,'distance':1.0/row.counts})
        GI.add_edge(row.sdistrict_id,row.hdistrict_id,{'weight':row.counts,'distance':1.0/row.counts})
    dics.append(nx.eigenvector_centrality(G,weight='weight'))
    names.append('eigen')
    
    dics.append(nx.eigenvector_centrality(GI,weight='weight'))
    names.append('righteigen')
    
    dics.append(nx.in_degree_centrality(G))
    names.append('indegree')
    
    dics.append(nx.out_degree_centrality(G))
    names.append('outdegree')
    
    dics.append(nx.closeness_centrality(G,distance='distance'))
    names.append('closeness')
    dics.append(nx.betweenness_centrality(G,weight='weight'))
    names.append('betweeness')
        
    def get(chunk):
        return chunk.counts.sum()
    dics.append(df.groupby('hdistrict_id').apply(get).to_dict())
    names.append('outgoing')
    dics.append(df.groupby('sdistrict_id').apply(get).to_dict())
    names.append('incoming')
    
    dic={}
    for n in nodes:
        neigh=G.neighbors(n)
        neigh.remove(n)
        N=len(neigh)
        s=0.0
        for i in range(len(neigh)):
            for j in range(i+1,len(neigh)):
                s+=G.has_edge(*(neigh[i],neigh[j]))
                s+=G.has_edge(*(neigh[j],neigh[i]))
        dic[n]=s/(N*(N-1))
    dics.append(dic)
    names.append('clustering')
    
    def check(row):
        return row.hdistrict_id==row.sdistrict_id
    df['check']=df.apply(check,1) 
    df2=df[df.check==False]
    dics.append(df2.groupby('hdistrict_id').apply(get).to_dict())
    names.append('outgoing_noself')
    dics.append(df2.groupby('sdistrict_id').apply(get).to_dict())
    names.append('incoming_noself')   
    for i in range(len(dics)):
        mapp(dics[i],names[i])
    ddf=pd.DataFrame.from_dict(ddf,orient='index') 
    ddf['district_id']=ddf.index
    return ddf


    
def read_population(omit=True):
    pop = pd.read_csv('../data/istanbul/population.csv')
    pop.rename(columns ={'ID':'district_id','Population 2015':'population'},inplace=True)
    pop = pop[['district_id','population']]
    if omit:
        pop = pop[~pop.district_id.isin(EXCLUDE)]
    pop.index = pop.district_id
    pop.drop(columns=['district_id'],inplace=True)
    return pop

def get_population_density(omit):
    pop = read_population(omit)
    df = pd.read_csv('../data/istanbul/Istanbul_district_area.txt')
    df.rename(columns={'area_km2':'area'},inplace=True)
    
    df2 = df.join(pop,on='district_id',how='left')
    if omit:
        df2 = df2[~df2.district_id.isin(EXCLUDE)]
    df2['popdse']=df2.population/df2.area
    return df2
def read_gdp():
    df = pd.read_csv('../data/istanbul/gdp.csv')
    for col in ['y2014','y2015','y2016']:
        df['log_{}'.format(col)] = np.log(df[col])
    return df
def get_gdp_growth(shopID2areaID):
    df2 = read_gdp()
    df2['delta1415']=(df2['y2015']-df2['y2014'])
    df2['delta1415p']=(df2['y2015']-df2['y2014'])/(df2['y2014'])*100
    df2['delta1516']=(df2['y2016']-df2['y2015'])
    df2['delta1516p']=(df2['y2016']-df2['y2015'])/(df2['y2015'])*100
    df2['delta1416']=(df2['y2016']-df2['y2014'])
    df2['delta1416p']=(df2['y2016']-df2['y2014'])/(df2['y2014'])*100
    return df2

def get_hpi():
    df=pd.read_csv('../data/istanbul/yearly_rent.csv')
    df.rename(columns={'rent_per_m2':'hpi'},inplace=True)
    #df=df[df.year==2014]
    return df

def create_dir(location):
    if not os.path.exists(location):
        os.makedirs(location)
        
class Result(object):
    
    def plot_boxplot(self,df,col,title,f_name):
        markers = dict(markerfacecolor='r', marker='o')
        plt.figure()
        plt.boxplot(df[col].values,flierprops=markers)
        plt.title(title)
        self.save_plot('box_{}'.format(f_name))
        
    def plot_predicted(self,df,x,y,x_lab,y_lab,f_name):
        if not f_name:
            f_name = '{}_{}.png'.format(x,y)
        plt.figure()
        if not x_lab:
            x_lab = x
        if not y_lab:    
            y_lab = y
        plt.subplot(111)
        plt.scatter(df[x].values, df[y].values,color='#1B5F68')
        
        plt.xlabel(x_lab,size=15)
        plt.ylabel(y_lab,size=15)
       
        maxx =max(df[x].max(),df[y].max())
        minn =min(df[x].min(),df[y].min())
        plt.plot([minn,maxx],[minn,maxx])
        plt.tight_layout()
        self.save_plot('scatter_{}'.format(f_name))
        
    def write_regression_result(self,res,f_name):
        result_location = '../results/{}/regressions/indv_results'.format(self.name)
        create_dir(result_location)
        with open(os.path.join(result_location,'{}.tex'.format(f_name)), 'w') as f:
            f.write(res.summary().as_latex())
    
    def write_regressions_summary(self,reses,model_names,reg_order,f_name):

        print (model_names,reg_order)
        info_dict = {'Observations':lambda x: "{0:d}".format(int(x.nobs)),}
                     #'R2':lambda x: "{:.4f}".format(x.rsquared),
                     #'Adjusted R2':lambda x: "{:.4f}".format(x.rsquared_adj)}   
        dfoutput = summary_col(reses,stars=True,info_dict=info_dict,model_names=model_names,regressor_order=reg_order)
        result_location = '../results/{}/regressions/summary'.format(self.name)
        create_dir(result_location)
        self.reg_table = dfoutput
        with open(os.path.join(result_location,'{}.tex'.format('{}_regression_summary_table'.format(f_name))), 'w') as f:
            f.write(dfoutput.as_latex())
    
    def plot_bar(self,df,col,ticks,title,name_suffix):
        colors = ['#EF476F','#00CFC1','#086788','#FF9F1C','#EB5E28']
# =============================================================================
#         colors = ['#086788','#00CFC1','#086788','#FF9F1C','#EB5E28']
# =============================================================================
        colors = ['#086788','#EF476F','#EF476F','#FF9F1C','#EB5E28']#FF9F1C'or #086788 b #00CFC1teal  #EF476F red
        color = colors[df.columns.get_loc(col)%len(colors)]
        
        
        y_pos = np.arange(len(ticks))
        values = df[col].values
        
        plt.figure()
        plt.bar(y_pos, values, align='center', alpha=0.5,color=color)
        plt.xticks(y_pos, ticks, rotation=270)
# =============================================================================
#         plt.ylabel(y_lab)
# =============================================================================
        plt.title(title)
        plt.tight_layout()
        self.save_plot('bar_{}.png'.format(name_suffix))
    
    def plot_heatmap(self,matrix,cmap,tick_label_x,tick_label_y,bar_label,name_suffix):
        plt.figure()
        sns.heatmap(matrix,cmap=cmap,cbar_kws={'label': bar_label})
        plt.yticks(np.arange(len(tick_label_y)),tick_label_y,size=8)
        plt.xticks(np.arange(len(tick_label_y)),tick_label_x,rotation=270,size=8,verticalalignment='top')
        plt.tight_layout()
        self.save_plot('heatmap_.png'.format(name_suffix))
        
    def run_regressions(self,df_indexes,cols,combinations,y,model_names,reg_order,f_name=None):
        reses = []
        assert len(df_indexes) == len(combinations)
        for i in range(len(combinations)):
            df = self.df[df_indexes[i]]
            combination = combinations[i]
            xs = [cols[c] for c in combination]
            reses.append(self.regress(df,xs,y))
        if not f_name:
            f_name = y
        self.write_regressions_summary(reses,model_names,reg_order,f_name)
            
        
    def run_glm(self,df,xs,y,family,f_name=None):
        if family =='gaussian':
            fam = sm.families.Gaussian()
        x_string='+'.join(xs)
        model = smf.glm('{} ~ {}'.format(y,x_string), data=df,family=fam)
        result = model.fit()

        if not f_name:
            f_name = 'glm_{}_{}_{}'.format(family,y,x_string) 
        self.write_regression_result(result,f_name)
        
        self.plot_residuals(df[y].values,result.fittedvalues,y,f_name=f_name)
        return result
        
        
    def regress(self,df,xs,y,f_name=None):

        x_string='+'.join(xs)
        mod = smf.ols('{} ~ {}'.format(y,x_string), data=df)
        res = mod.fit()

        if not f_name:
            f_name = '{}_{}'.format(y,x_string) 
        self.write_regression_result(res,f_name)
        
        self.plot_residuals(df[y].values,res.fittedvalues,y,f_name='{}'.format(f_name))
        return res
    
    def plot_residuals(self,y,y_fitted,y_name,f_name):
        df=pd.DataFrame()
        
        df.loc[:,'errors'] =  y - y_fitted
        df.loc[:,'y'] = y
        df.loc[:,'y_fitted'] = y_fitted
        y_name = 'Values'
        self.plot_scatter(df,'y_fitted','errors',x_lab ='Fitted {}'.format(y_name),y_lab='Residuals',f_name='resid_{}'.format(f_name))
        
                
    def save_plot(self,fig_name):
        plot_location = '../results/{}/plots'.format(self.name)
        create_dir(plot_location)
        plt.savefig(os.path.join(plot_location,'{}.png'.format(fig_name)),dpi=300)
    
    def get_corr(self,df,x,y):
        return df[[x, y]].corr()[y][x]
    
    def plot_scatter(self,df,x,y,x_lab=None,y_lab=None,has_best_fit=False,controls=None,f_name=None,**kwargs):
        if not f_name:
            f_name = '{}_{}.png'.format(x,y)
        plt.figure()
        if not x_lab:
            x_lab = x
        if not y_lab:    
            y_lab = y
        plt.subplot(111)
        plt.scatter(df[x].values, df[y].values,color='#1B5F68')
        
        plt.xlabel(x_lab,size=15)
        plt.ylabel(y_lab,size=15)
        if has_best_fit:
            corr=self.get_corr(df,x,y)
            if 'corr_loc' not in kwargs:
                corr_loc_x = df[x].mean()
                corr_loc_y = df[y].mean()
            else:
                corr_loc_x = kwargs['corr_loc'][0]
                corr_loc_y = kwargs['corr_loc'][1]
            plt.text(corr_loc_x,corr_loc_y,'Correlation : {}'.format(np.round(corr,4)),size=15)
            plt.plot(np.unique(df[x].values), np.poly1d(np.polyfit(df[x].values, df[y].values, 1))(np.unique(df[x].values)))
        else:
            corr=0
        plt.tight_layout()
        self.save_plot('scatter_{}'.format(f_name))
        
        return corr
    def plot_bubble_map(self,df,col,scale,u_districts,name):
        
        if 'beijing' in self.name:
            
            loc = [39.9042, 116.4074]
            state_geo = '../data/beijing/bj_shapefile/bj_shapefile.geojson'
            district_id_col='districtID'
            
        elif 'usa' in self.name:
            df = df.copy()
            district_id_col = 'GEOID'
            state_geo = '../data/usa/shapefiles/cb_2016_us_county_20m/usa_shapefile.geojson'
            loc = [37.0902, -95.7129]
            
        else:
            loc = [41.0082, 28.9784]
            state_geo = '../data/istanbul/district_level_shape/district.geojson'
            district_id_col='districtid'
        style_function = lambda x: {'fillOpacity': 0.15 if       
                                x['properties'][district_id_col] in u_districts else
                                 0.85, 'fillColor' : "#000000",'stroke': False }
        mapbg=folium.GeoJson(state_geo,style_function)
        # Make an empty map
        m = folium.Map(location=loc, tiles="cartodbpositron", zoom_start=10)
         
        # I can add marker one by one on the map
        for i in range(0,len(df)):
           folium.Circle(
              location=[df.iloc[i]['lat'], df.iloc[i]['lon']],
# =============================================================================
#               popup=df.index[i],
# =============================================================================
              radius=df.iloc[i][col]*scale,
              color='crimson',
              fill=True,
              fill_color='crimson'
           ).add_to(m)
        mapbg.add_to(m)
        # Save it as html
        location = '../results/{}/maps'.format(self.name)
        create_dir(location)
        m.save(os.path.join(location,'bubble_map_{}.html'.format(name)))

    def plot_map(self,df,col,legend_string,name):
        # Initialize the map:
        if 'beijing' in self.name:
            state_geo = '../data/beijing/bj_shapefile/bj_shapefile.geojson'
            loc = [39.9042, 116.4074]
            district_id_col='districtID'
            
        elif 'usa' in self.name:
            df = df.copy()
            district_id_col = 'GEOID'
            state_geo = '../data/usa/shapefiles/cb_2016_us_county_20m/usa_shapefile.geojson'
            loc = [37.0902, -95.7129]
            df.district_id = df.district_id.astype(str)

        else:
            
            state_geo = '../data/istanbul/district_level_shape/district.geojson'
            loc = [41.0082, 28.9784]
            district_id_col='districtid'
            style_function = lambda x: {'fillOpacity': 1 if       
                                x['properties'][district_id_col]==0 else
                                 0, 'fillColor' : "#000000",'stroke': False }
            mapbg=folium.GeoJson(state_geo,style_function)
        

        m = folium.Map(location=loc, zoom_start=10,tiles='cartodbpositron')
         
        # Add the color for the chloropleth:

    

        
        m.choropleth(
         geo_data=state_geo,
         name='choropleth',
         data=df,
         columns=['district_id',col],
         key_on='feature.properties.{}'.format(district_id_col),
         nan_fill_opacity=0.6,
         fill_color='YlGnBu',
         fill_opacity=0.7,
         line_opacity=0.2,
         legend_name=legend_string
        )
        if 'beijing' not in self.name and 'usa' not in self.name:
            
            mapbg.add_to(m)
        # Save to html
        location = '../results/{}/maps'.format(self.name)
        create_dir(location)
        m.save(os.path.join(location,'choro_map_{}.html'.format(name)))
        
class CommoditiesAndFlow(Result):
    
    name = 'comm_and_flow'
    
    def __init__(self):
        self.df = {}
        self.df_commodities = read_availablity_diversity()
        self.df_flow_diversity = read_in_out_diversity()
        self.df['istanbul'] = self.df_commodities.join(self.df_flow_diversity,on='district_id',how='left')
    
    def plot_results(self):
        
        df = pd.DataFrame()
        xs = self.df_commodities.columns
        ys = self.df_flow_diversity.columns
        
        for x in xs:
            for y in ys:

                corr=self.plot_scatter(self.df['istanbul'],x,y,has_best_fit=True)
                df=df.append({'x':x,'y':y,'correlation':corr},ignore_index=True)
        df.sort_values('correlation',ascending=False,inplace=True)
        loc = '../results/{}'.format(self.name)
        create_dir(loc)
        df.to_csv(os.path.join(loc,'corr_results.csv'),index=False)
        
    def run_for_results(self):
        self.plot_results()
        
        
class FlowAndEconomicOutput(Result):
    name = 'flow_and_econ'
    
    def __init__(self):
        self.df = {}
        flow = get_flow_df(is_repeat=True)
        pop = read_population()
        flow = flow.join(pop,on='district_id',how = 'left')
        flow['totalflow'] = flow.inflow+flow.outflow
        df= read_gdp()
        self.df['istanbul'] = df.join(flow,on='district_id',how="left")
        
    def plot_results(self):
        
        plt.figure()
    #==============================================================================
    #     xlab='Number of unique shop categories'
    #==============================================================================
        x_lab= 'Diversity of People entering in district'
        y_lab= "Diversity of Flow (in Thousands)"
        y_lab= "log(GDP)"
        x_lab='Volume of Total Flow'
        x_lab = None
# =============================================================================
#         y_lab = None
# =============================================================================
        xs = ['inflow','outflow','totalflow']
        ys = ['log_{}'.format(c) for c in ['y2014','y2015','y2016']]
        ys = ['log_{}'.format(c) for c in ['y2014']]
# =============================================================================
#         ys = ['y2014']
# =============================================================================
        df = pd.DataFrame()
        for x in xs:
            for y in ys:
                corr=self.plot_scatter(self.df['istanbul'],x,y,x_lab,y_lab,has_best_fit=True,corr_loc=(51000,18))
                df=df.append({'x':x,'y':y,'correlation':corr},ignore_index=True)
        df.sort_values('correlation',ascending=False,inplace=True)
        loc = '../results/{}'.format(self.name)
        create_dir(loc)
        df.to_csv(os.path.join(loc,'corr_results.csv'),index=False)
    def get_regressions(self):
        df_index = ['istanbul']*4
        cols = ['inflow','outflow','withinflow','population']
        combinations = [(0,1,),(0,1,2),(0,1,3),(0,1,2,3)]
        ys = ['y2014','y2015','y2016']
        ys = ys + ['log_{}'.format(y_string) for y_string in ys]
        for y in ys:
            self.run_regressions(df_index,cols,combinations,y,['Model 1','Model 2','Model 3','Model 4'],[],'flow_vs_gdp')
        
    def run_for_results(self):
        self.plot_results()
        self.get_regressions()

def read_income_spend():
    df = pd.read_csv('../data/istanbul/cc_vs_hh.csv')
    df = df.rename(columns = {'Average household income for each district':'avg_income','Average transaction per customer sample in each district':'avg_spending'})
    return df
def read_areas():
    df = pd.read_csv('../data/istanbul/Istanbul_district_area.txt')
    df = df[~df.district_id.isin(EXCLUDE)]
    df.sort_values(by='district_id',inplace=True)
    return df
    
def read_poi():
    poi = pd.read_csv('../data/istanbul/attractiveness.csv')
    poi.rename(columns={'POLYGON_NM':'district_name'},inplace=True)
    to_remove = ['MajHwys','AutoSvc','CommSvc']
# =============================================================================
#     to_remove = ['MajHwys','SecHwys','Parking','RailRds']
# =============================================================================
# =============================================================================
#     ['TrnsHubs','Hospitals','EduInst','FinInst','AutoSvc','CommSvc']
# =============================================================================
# =============================================================================
#          ['ParkRec','Rstrnts','Entrtnmt','Business','Shopping','TrvDest']
# =============================================================================
        
    poi_cols = poi.columns.tolist()  
    for o in ['district_name','district_id','poidensity','divcount','POI_diversity','POI_sum']+to_remove:
        poi_cols.remove(o)
    
    poi.loc[:,'poi_diversity_0'] = poi.apply(lambda row: scipy.stats.entropy(row[poi_cols].values.tolist()),1)
    poi.loc[:,'poi_sum_0'] = poi.apply(lambda row: row[poi_cols].sum(),1)
# =============================================================================
#     for i in range(len(to_remove)):
#         poi_cols.remove(to_remove[i])
#         poi.loc[:,'poi_diversity_{}'.format(i+1)] = poi.apply(lambda row: scipy.stats.entropy(row[poi_cols].values.tolist()),1) 
#         poi.loc[:,'poi_sum_{}'.format(i+1)] = poi.apply(lambda row: row[poi_cols].sum(),1)
# =============================================================================
    poi.sort_values(by='district_id',inplace=True)
    poi.index=poi.district_id
    return poi

def get_distance_ij():
    dist = pd.read_csv('../data/istanbul/effective_distance.csv').rename(columns={'FID':'district_id','POLYGON_NM':'district_name'})
    dist.index=dist.district_id
    name2id = dist[['district_id','district_name']]
    name2id.index = name2id.district_name
    name2id = name2id['district_id']
    dist.drop(columns = ['X','Y','district_id','district_name'],inplace=True)
    dist.rename(columns={i:name2id.at[i] for i in dist.columns},inplace=True)
    return dist


def get_total_flows(is_repeat):
    flow = get_flow_matrix(is_repeat)
    #df = pd.merge(poi[['district_id','POI_sum','POI_diversity']],dist[])
    flow.rename(columns = {'hdistrict_id':'district_i','sdistrict_id':'district_j'},inplace=True)
    workflow = get_work_flow()
    workflow.rename(columns = {'hdistrict_id':'district_i','wdistrict_id':'district_j'},inplace=True)
    districts = flow.district_i.unique()
    all_ij = []
    for i in districts:
        for j in districts:
            all_ij.append('{}_{}'.format(i,j))
    flow['id'] = flow.district_i.astype(str) + '_' + flow.district_j.astype(str)
    workflow['id'] = workflow.district_i.astype(str) + '_' + workflow.district_j.astype(str)
    workflow.drop(columns=['district_j','district_i'],inplace=True)
    totalflow = pd.merge(flow,workflow,on='id',how='outer')
    mask = totalflow['workflow'].isnull()
    totalflow.loc[mask,'workflow'] = 0
    mask = totalflow['flow'].isnull()
    totalflow.loc[mask,'flow'] = 0
    missing = pd.DataFrame.from_dict({'district_i':[15],'district_j':[86],'id':['15_86'],'workflow':[0],'flow':[0]})
    totalflow = totalflow.append(missing)
    
    totalflow['totalflow'] = totalflow['flow']+ totalflow['workflow']
    
    mask = totalflow['district_i'].isnull()
    
    def get_i_or_j(string,ij):
        if ij == 'i':
            idx = 0
        elif ij == 'j':
            idx = 1
        return int(string.split('_')[idx])
    totalflow.loc[mask,'district_i'] = totalflow[mask].id.apply(lambda x:get_i_or_j(x,'i'))
    totalflow.loc[mask,'district_j'] = totalflow[mask].id.apply(lambda x:get_i_or_j(x,'j'))
    exisiting_ij = totalflow.id.unique()
    for ij in all_ij:
        if ij not in exisiting_ij:
            print('{} does not exist'.format(ij))
    
    
    return totalflow
def normalize_amenities(poi):
    amenities = poi.columns.tolist()
    for o in ['district_name','district_id','poidensity','divcount','POI_diversity','POI_sum']:
        amenities.remove(o)
    def get_norm_entro(row,amenities):
        entro = scipy.stats.entropy(row[amenities].astype(int).values)
        norm = np.ones(len(amenities))
        return entro/scipy.stats.entropy(norm)
    poi['poi_diversity_norm'] = poi.apply(lambda x: get_norm_entro(x,amenities),1)
    poi['poi_sum_norm'] = poi.POI_sum / poi.POI_sum.max()
    return poi
    
    
def add_poi_measure(row,measure,poi):
    i = row.district_i
    j = row.district_j
    return  poi.loc[i,measure],poi.loc[j,measure]

def add_distance_measure(row,dist):
    return dist.loc[row.district_i,row.district_j]


class HuffModel(Result):
    REMOVE_RANGE = 1
    name = 'huffmodel'
    def __init__(self):
        totalflow = get_total_flows(is_repeat=True)       
        dist = get_distance_ij()
        poi = read_poi()
# =============================================================================
#         poi = normalize_amenities(poi)
# =============================================================================
        
        for n in range(self.REMOVE_RANGE):
            totalflow['poi_sum_{}_i'.format(n)],totalflow['poi_sum_{}_j'.format(n)] = zip(*totalflow.apply(lambda x:add_poi_measure(x,'poi_sum_{}'.format(n),poi),1))
            totalflow['poi_diversity_{}_i'.format(n)],totalflow['poi_diversity_{}_j'.format(n)] = zip(*totalflow.apply(lambda x:add_poi_measure(x,'poi_diversity_{}'.format(n),poi),1))
# =============================================================================
#         totalflow['poi_sum_norm_i'],totalflow['poi_sum_norm_j'] = zip(*totalflow.apply(lambda x:add_poi_measure(x,'poi_sum_norm',poi),1))
#         totalflow['poi_diversity_norm_i'],totalflow['poi_diversity_norm_j'] = zip(*totalflow.apply(lambda x:add_poi_measure(x,'poi_diversity_norm',poi),1))
# =============================================================================
        totalflow['distance_ij'] = totalflow.apply(lambda x: add_distance_measure(x,dist),1)      
        x_cols = ['distance_ij']
        for n in range(self.REMOVE_RANGE):
            x_cols.append('poi_sum_{}_j'.format(n))
            x_cols.append('poi_diversity_{}_j'.format(n))
        
        for col in x_cols:
            totalflow['log_{}'.format(col)] = np.log(totalflow[col])
        self.df = totalflow
        self.poi = poi
        self.dist = dist
    
    def plot_flow_diversity(self):
        df = self.df
        df['totalflow_thou'] = df.totalflow/1000
        self.plot_scatter(df,'poi_diversity_0_j','totalflow_thou','Amenities Diversity', 'Total Volume of Flow (in Thousands)',has_best_fit=True,f_name='div_vs_flow')
    def run_for_results(self):
        self.plot_poi_bars()
        self.plot_flow_heatmap()
        self.plot_distance_heatmap()
        self.plot_flow_diversity()
        self.run_glms()
        self.run_ols()
        self.plot_indiv_fits()
    
    def add_glm_results(self,df_summary,y_series,reses):
        for res in reses:
            error = (y_series - reses[res].fittedvalues) 
            mean_square_error = (error*error).mean()
            abs_error = abs(error).mean()
            pseudo_r2 = 1-(reses[res].deviance/reses[res].null_deviance)
            series = pd.Series({
                'Model': res,
                'Root Mean Squared Error' : np.sqrt(mean_square_error),
                'Mean Absolute Error': abs_error,
                'Pseudo R2': pseudo_r2,
                'Residual Deviance': reses[res].deviance,
                'Df Residual': reses[res].df_resid,
                })
            series = series.append(reses[res].params)
            series.rename({'log_distance_ij':'Gamma','log_poi_diversity_0_j':'Beta','log_poi_sum_0_j':'Alpha'},inplace=True)
            series.rename({'log_geo_distance_ij':'Gamma','log_geo_poi_diversity_0_j':'Beta','log_geo_poi_sum_0_j':'Alpha'},inplace=True)
            df_summary = df_summary.append(series,ignore_index=True)
        return df_summary
    def glm(self,y,xs,f_name=None):
        df = self.df
        x_string='+'.join(xs)
        
        model_poisson = smf.glm('{} ~ {}'.format(y,x_string), data=df,family=sm.families.Poisson())
        res_poisson = model_poisson.fit()
        
        model_nb = smf.glm('{} ~ {}'.format(y,x_string), data=df,family=sm.families.NegativeBinomial())
        res_nb = model_nb.fit(method='newton')
        
        if not f_name:
            f_name = '{}_{}'.format(y,x_string) 
        self.write_regression_result(res_nb,'nb_{}'.format(f_name))
        self.write_regression_result(res_poisson,'pois_{}'.format(f_name))
        self.plot_residuals(df[y], res_nb.fittedvalues, 'Flow Counts', 'nb_{}'.format(f_name))
        self.plot_residuals(df[y], res_poisson.fittedvalues, 'Flow Counts', 'poisson_{}'.format(f_name))
        
        
        #### comparing mse
        df_summary = pd.DataFrame()
        reses = {'Poisson':res_poisson, 'Negative Binomial':res_nb}
        df_summary = self.add_glm_results(df_summary,df[y],reses)
        
        location = '../results/{}/regressions/summary'.format(self.name)
        create_dir(location)
        df_summary.to_csv(os.path.join(location,'model_fit_summary.csv'),index=False)
            
        return res_poisson,res_nb

     
    def run_glms(self):
        
        for n in range(self.REMOVE_RANGE):
            x_cols = ['poi_sum_{}_j'.format(n),'poi_diversity_{}_j'.format(n),'distance_ij']
            y = 'totalflow'
            xs = ['log_{}'.format(x) for x in x_cols]
# =============================================================================
#             xs_control = ['log_{}'.format(x) for x  in ['distance_ij']]
# =============================================================================
            pois,nb = self.glm(y,xs,str(n))
# =============================================================================
#             pois_control,nb_control = self.run_glm(y,xs_control,'control_{}'.format(n))
# =============================================================================
            
 
    def plot_indiv_fits(self):
        location = '../results/{}/regressions/parameters'.format(self.name)
        df_names = pd.read_csv('../data/istanbul/Istanbul_district_area.txt')[['district_id','district_name']]
        
        df_summary = pd.read_csv(os.path.join(location,'parameters_0.csv'))
        df_summary = df_summary[df_summary.district_id!='average'].copy()
        df_summary.district_id = df_summary.district_id.astype(int)
        df_summary['log_geo_distance_ij'] = - df_summary['log_geo_distance_ij']
        df_summary = pd.merge(df_summary,df_names,on='district_id',how='left')
        cols = {'log_geo_poi_sum_0_j': 'Alpha','log_geo_poi_diversity_0_j':'Beta','log_geo_distance_ij':'Gamma','r2':'R Squared'}
        
        for col in cols:
            self.plot_bar(df_summary, col, df_summary.district_name, cols[col], cols[col])
        
        
    def run_ols(self):
                    
        def get_r2(chunk):
            df_norm = chunk.copy()
            error = (df_norm['totalflow_not_norm'] - df_norm['fitted_flow']) 
            mse = (error*error).mean()
            yavg = df_norm.totalflow_not_norm.mean()
            error2 = df_norm['totalflow_not_norm']-yavg
            mse2 = (error2*error2).mean()
            return 1-(mse/mse2)
        
        def normalize_prob(chunk):
            chunk = chunk.copy()
            norm = chunk.prob.sum()
            chunk.loc[:,'prob'] = chunk.prob/norm
            chunk.loc[:,'fitted_flow'] = chunk.prob * (chunk.totalflow_not_norm.sum())
            return chunk


        regs_global = []
        for n in range(self.REMOVE_RANGE):
            

            df = self.df.copy()
            xs = ['poi_sum_{}_j'.format(n),'poi_diversity_{}_j'.format(n),'distance_ij']        
            def normalize_geo_log(chunk,xs):
                
                chunk = chunk.copy()
                haszero = ((chunk.totalflow == 0).sum()) > 0
                if haszero:
                    chunk = chunk[chunk.totalflow!=0]
                chunk.loc[:,'totalflow_not_norm'] = chunk.totalflow.copy()
                summ = chunk.totalflow.sum()
                chunk.totalflow = chunk.totalflow/summ
                geo= {}
                for var in xs+ ['totalflow']:
                    geo[var] = scipy.stats.gmean(chunk[var])
                    chunk['log_geo_{}'.format(var)] = np.log(chunk[var]/geo[var])
                for var in geo:
                    chunk.loc[:,'geo_{}'.format(var)]=geo[var]
                return chunk
            df_norm = df.groupby('district_i').apply(lambda x:normalize_geo_log(x,xs)).reset_index(drop=True)
            #need to deal with zeros
            y = 'log_geo_totalflow'
            xs2 = ['log_geo_{}'.format(x) for x in xs]
            
        
            df_params=pd.DataFrame()
            districts = []
            regs=[]
            r2s=[]
            for i, chunk in df_norm.groupby('district_i'):
                chunk = chunk.copy()
                reg = self.regress(chunk,xs2,y,f_name='huff_ols_{}'.format(n))
                chunk['fitted_val']= reg.fittedvalues
                chunk['prob'] = np.exp(reg.fittedvalues)
                chunk = normalize_prob(chunk)
                r2s.append(reg.rsquared)
                df_params = df_params.append(pd.DataFrame(reg.params).transpose())
                districts.append(int(i))
                regs.append(reg)
            
            df_params = df_params.append(pd.DataFrame(df_params.mean()).transpose())
            districts.append('average')
            df_params['district_id'] = districts
            r2s.append(np.mean(r2s))
            df_params['r2'] =r2s
            self.write_regressions_summary(regs,[i for i in range(len(regs))],['Observations'],'huff_{}'.format(n))
            parameters_loc = '../results/{}/regressions/parameters'.format(self.name)
            create_dir(parameters_loc)
            df_params.to_csv(os.path.join(parameters_loc,'parameters_{}.csv').format(n))

            reg = self.regress(df_norm,xs2,y,f_name='huff_ols_global_{}'.format(n))
            regs_global.append(reg)
            
        
            df_norm.loc[:,'prob']=np.exp(reg.fittedvalues)
            df_norm['fitted_val']=reg.fittedvalues
            df_norm = df_norm.groupby('district_i').apply(normalize_prob).reset_index(drop=True)
            
            
            
                
            
            error = (df_norm['totalflow_not_norm'] - df_norm['fitted_flow']) 
            mean_square_error = np.sqrt((error*error).mean())
            
            mse = (error*error).mean()
            yavg = df_norm.totalflow_not_norm.mean()
            error2 = df_norm['totalflow_not_norm']-yavg
            mse2 = (error2*error2).mean()
            
            print('r2',1-(mse/mse2))
            abs_error = abs(error).mean()
            for s in ['totalflow_not_norm','fitted_flow']:
                df_norm['log_{}'.format(s)]=np.log(df_norm[s])
            self.plot_predicted(df_norm,'log_totalflow_not_norm','log_fitted_flow','log(Observed Flow)','log(Predicted Flow)',f_name='obs_vs_pred')


            
            location = '../results/{}/regressions/summary'.format(self.name)
            df_summary = pd.read_csv(os.path.join(location,'model_fit_summary.csv'))
            result_glm = self.run_glm(df_norm,xs2,y,family='gaussian')
            df_summary = self.add_glm_results(df_summary, df_norm[y], {'Gaussian':result_glm})
            df_summary.index = df_summary.Model
            df_summary.loc['Gaussian','Mean Absolute Error'] = abs_error
            df_summary.loc['Gaussian','Root Mean Squared Error'] = mean_square_error
                

            
            col_arranged = ['Model','Root Mean Squared Error','Pseudo R2', 'Residual Deviance','Df Residual', 'Intercept', 'Alpha','Beta', 'Gamma']
            df_summary = df_summary[col_arranged]
            df_summary.to_csv(os.path.join(location,'model_fit_summary.csv'),index=False)
            
            print(df_summary.to_latex(index=False),file=open(os.path.join(location,'model_fit_summary_latex.tex'),'w'))
# =============================================================================
#         self.write_regressions_summary(regs_global,'huff_globals'
# =============================================================================
        
        self.df_norm=df_norm
        
    def plot_poi_bars(self):
        title_mapper ={'AutoSvc' : 'Auto Services',
                       'Business' : 'Businesses',
                       'CommSvc' : 'Community Services',
                       'EduInst' : 'Education Institutes',
                       'Entrtnmt' : 'Entertainment Venues',
                       'FinInst': 'Financial Institutes', 
                       'Hospitals': 'Hospitals',
                       'ParkRec' : 'Recreational Parks',
                       'RailRds': 'Railroads', 
                       'Rstrnts': 'Restaurants', 
                       'Shopping' : 'Shopping Venues', 
                       'TrnsHubs': 'Transportion Hubs',
                       'TrvlDest': 'Travel Destinations', 
                       'poi_diversity_0': 'POI Diversity (Entropy)', 
                       'poi_sum_0': 'POI Sum'
            }
        cols = self.poi.columns.to_list()
        omit = ['district_name','district_id','poidensity','divcount', 'POI_sum', 'POI_diversity']
        
    
        for o in omit:
            cols.remove(o)
        for col in cols:
            if col in title_mapper:
                title = title_mapper[col]
            else:
                title = col
            self.plot_bar(self.poi,col,self.poi.district_name,title,col)

        
    def plot_distance_heatmap(self):
        matrix = np.log(self.dist.values)
        ticks = self.poi.district_name.values
        self.plot_heatmap(matrix,'PuBu',ticks,ticks,'Travel Time (minutes)','distance')
        
    def plot_flow_heatmap(self):
        df_info=read_areas()
        districts=df_info.district_id.values
        df = get_flow_matrix(is_repeat=True)
    
        mapper={}
        demapper={}
        matrix=np.zeros(shape=(len(districts),len(districts)))
        for i in range(len(districts)):
            demapper[i]=districts[i]
            mapper[districts[i]]=i
        
        def addToMat(row):
            try:
                fromm=mapper[row.hdistrict_id]
                to=mapper[row.sdistrict_id]
# =============================================================================
#                 if to!=fromm:
# =============================================================================
                matrix[fromm,to]=np.log(row.flow)
            except:
                pass
        df.apply(addToMat,1)
        
        self.plot_heatmap(matrix,'PuBu',df_info.district_name.values,df_info.district_name.values,'log(Flow)','flow')
# =============================================================================
#  |       sns.heatmap(df_diverse.mcc_detailed.values.reshape(39,1),cmap='PuBu',cbar_kws = dict(use_gridspec=False,location="top"))
# =============================================================================

class Supplementary(Result):
    name = 'supplementary'

    def __init__(self):
        self.df ={}
        self.df['pop'] = read_population(omit=False)
        self.df['rep'] = population_representation(omit=False)   
        self.df['area']= read_areas() 
        df = get_population_density(omit=True)
        df = pd.merge(df,self.df['area'][['district_id']],on='district_id',how='left')
        df.sort_values(by='district_id',inplace=True)
        self.df['pop_36'] = df
        self.df['hpi']=get_hpi()
        self.df['income_spending'] = read_income_spend()
    def get_population_plots(self):
        df = self.df['pop'].join(self.df['rep'],how='left')
        df['district_id'] = df.index
        self.plot_map(df,'population','Population (in thousands)','population')
        self.plot_map(df,'pop_rep','Sample counts','sample')
        self.plot_scatter(df,'pop_rep','population','Sample Size','Population Size',has_best_fit=True,corr_loc= (500,50))
        
        df = self.df['pop_36']
        self.plot_bar(df, 'population', df.district_name.values,'Population (in thousands)','population')
        self.plot_bar(df, 'popdse', df.district_name.values,'Density (in thousands/km2)','population_density')
        
    def get_area_plots(self):
        df = self.df['area']
        self.plot_map(df,'area_km2','Area (in km2)', 'area')
        self.plot_bar(df,'area_km2',df.district_name,'Area (in km2)','area')
        
    def get_rent_plots(self):
        df = self.df['hpi']
        self.plot_map(df,'hpi','Rent Price','rent')
        
    def plot_income_spending(self):
        df = self.df['income_spending']
        self.plot_scatter(df,'avg_spending','avg_income','Average Transaction','Average Income',True,f_name='cc_vs_hh',corr_loc=(8000,25))
        
    def run_for_results(self):
        self.get_population_plots()
        self.get_area_plots()
        self.get_rent_plots()
        self.plot_income_spending()
        
        
def get_bj_population_rent():
    df=pd.read_csv('../data/beijing/bj_pop.csv')[['districtID','Popu','Popu_float','Pop_Sum','Dens_Pob','Area_km2']]
    df.rename(columns={'districtID':'district_id','Pop_Sum':'population','Popu_float':'population_float','Popu':'population_res','Area_km2':'area_km2','Dens_Pob':'popdse'},inplace=True)
    df['population']=df['population']/1000
    df['popdse'] = df['popdse']/1000
    
    df2=pd.read_csv('../data/beijing/bj_house_price.csv')
    df2.rename(columns={'districtID':'district_id',u'Mean House Price (MHP)':'hpi'},inplace=True)
    df=pd.merge(df,df2[['district_id','hpi']],on='district_id',how='left')

    return df

def get_bj_consumption_data():
     df = pd.read_csv('../data/beijing/beijing_consumption.csv',encoding='utf-8',sep='\t')
     df.rename(columns={'districtID': 'district_id','numberofCustomers':'pop_rep','dEigen':'eigen_centrality','categoryEntropy':'consumption_diversity'},inplace=True)
     return df[['district_id','eigen_centrality','pop_rep','economy']]
 
def get_coupon_shop():
    coupon_shop = pd.read_csv('../data/beijing/coupon_shop.csv')
    def strip(row):
        
        return row.coupon.split('/')[-1][:-5],row.shop.split('/')[-1]


    coupon_shop['coupon_id'],coupon_shop['shop_id']=zip(*coupon_shop.apply(strip,1))
    
    coupon_shop.drop_duplicates(subset=['shop_id','coupon_id'],inplace=True)
    num_shop = pd.DataFrame(coupon_shop.groupby('coupon_id')['shop_id'].nunique().rename('nshop'))
    num_shop['is_rejected'] = num_shop.nshop > 1
    coupon_shop = coupon_shop.join(num_shop[['is_rejected']],how='left',on='coupon_id')
    coupon_shop = coupon_shop[~coupon_shop.is_rejected].copy()
    
    
    return coupon_shop


def get_transactions():
    coup = []
    customer = []
    coupon= json.load(open('../data/beijing/bz_user_time.json', 'rb'))
    for dt in coupon:
        for key,value in coupon[dt].items():
            customer.append(key.split('\t')[1])
            coup.append(key.split('\t')[0])
    df = pd.DataFrame(np.stack([coup,customer],1),columns = ['coupon_id','customer_id'])
    
    
      
    return df

def get_bj_transactions():
    
    coupon_shop= get_coupon_shop()
    df = get_transactions()
    
    df = pd.merge(df,coupon_shop,how='left',on='coupon_id')
    df.dropna(subset=['shop_id'],inplace=True)
    return df

def get_bj_shop_info():
    shop = json.load(open('../data/beijing/shop.json', 'rb'))
    shop_df = pd.DataFrame.from_dict(shop,orient='index')[['num_sales','value_saled','category']].rename(columns={'value_saled':'sales_value'})
    

    shop_df_dis= pd.read_csv('../data/beijing/meituan_district_mapping.csv').rename(columns={'meituan_id':'shop_id','districtID':'district_id','districtName':'district_name'})
    shop_df_dis = shop_df_dis.drop(columns=['economy','districtProvince', 'districtRegion'])
    
    shop_df_dis.shop_id = shop_df_dis.shop_id.astype(str)
    
    shop_df = shop_df_dis.join(shop_df,how='left',on='shop_id')
    
    #extra info
# =============================================================================
#     shop_dp_mapping= pd.read_csv('../data/beijing/datawithAllfeatures_sep.csv',sep='\t')
# =============================================================================
    return shop_df



class SupplementaryChina(Result):
    name = 'supplementary_beijing'
    def __init__(self):
        self.df = pd.merge(get_bj_population_rent(),get_bj_consumption_data(),on='district_id',how='outer').dropna()
        self.shop = get_bj_shop_info()
        self.df_trans = get_bj_transactions()
        
    def get_population_plots(self):
        df = self.df
        self.plot_map(df,'population','Population (in thousands)','population')
        self.plot_map(df,'pop_rep','Sample counts','sample')
        self.plot_scatter(df,'pop_rep','population','Sample Size','Population Size',has_best_fit=True,corr_loc= (500,50))
        
        self.plot_bar(df, 'population', ['' for _ in range(len(df))],'Population (in thousands)','population')
        self.plot_bar(df, 'popdse', ['' for _ in range(len(df))],'Density (in thousands/km2)','population_density')
        
    def get_area_plots(self):
        df = self.df
        self.plot_map(df,'area_km2','Area (in km2)', 'area')
        self.plot_bar(df,'area_km2',['' for _ in range(len(df))],'Area (in km2)','area')
    
    def get_house_price_plots(self):
        df = self.df
        self.plot_map(df,'hpi','Housing Price','houseprice')
        
    def run_for_results(self):
        sns.set_style("whitegrid", {'axes.grid' : False})
        self.get_population_plots()
        self.get_area_plots()
        self.get_house_price_plots()
        sns.set_style("whitegrid", {'axes.grid' : True})
        self.plot_shop_categories()
        self.plot_trans_summary()
        self.plot_econ()
    def plot_shop_categories(self):
        df = pd.merge(self.df_trans,self.shop[['shop_id','category']],on='shop_id',how='left')
        cat_bar = pd.DataFrame(df.category.value_counts().rename('count'))
        cat_map = {
            '火锅': 'Hot Pot', '小吃快餐' : 'Fast Food', '甜点饮品': 'Dessert', '川湘菜': 'Szechuan Cuisine',
            '烧烤烤肉' : 'Barbeque', '日韩料理' : 'Japanese', '其他美食': 'Misc', '京菜鲁菜': 'Beijing Cuisine', '西餐': 'Western',
       '自助餐':'Buffet', '东北菜' : 'Dongbei Cuisine', '海鲜':'Seafood', '粤港菜': 'Hongkong Cuisine', '西北菜': 'Xibei Cuisine', 
       '江浙菜':'Jiangzhe Cuisine', '云贵菜': 'Yungui Cuisine', '新疆菜': 'Xinjiang cuisine' , '东南亚菜': 'South East Asian', '创意菜':'Fusion',
       '农家乐': 'Rural'}
        
        cat_bar.index = [cat_map[c] for c in cat_bar.index]
        
        self.plot_bar(cat_bar,'count',cat_bar.index.values,'Number of Transactions per Category','beijing_trans_cat_count')
    
    def plot_trans_summary(self):
        df = pd.merge(self.df_trans,self.shop[['shop_id','lat','lon','district_id']],on='shop_id',how='left')
        districts = df.district_id.unique()
        def get_trans_per_shop(chunk):
            return pd.Series({'lat':chunk.lat.iloc[0],'lon':chunk.lon.iloc[0],'count':1})
        trans_count = df.groupby('shop_id').apply(get_trans_per_shop)
        trans_count.dropna(subset=['lat','lon'],inplace=True)
        scale = 1
        self.plot_bubble_map(trans_count, 'count', scale,districts, 'trans_count')
        

        
        
    def plot_econ(self):
        df = pd.read_csv('../data/beijing/beijing_econ.csv')
        df['log_econ'] = np.log(df.economy)
        self.plot_map(df,'log_econ','Total Capital Assets (Log Scale)','econ')


class GrowthVsDiv(Result):
    
    name = 'growth'
    
    def __init__(self):
        df = {}
        df1 = pd.read_csv('../data/istanbul/istanbul_cc.csv')
        df1 = df1.rename(columns={'delta1416p':'growth','HPI':'hpi','weightedEigenCentralityDist':'geo_centrality','UA':'ua'})
        df['istanbul'] = df1
        df1 = pd.read_csv('../data/beijing/beijing_cc.csv')
        df1['log_econ'] = np.log(df1.economy)
        df1 = df1.rename(columns={'logEcondse':'growth','HPI':'hpi','dEigen':'geo_centrality','categoryEntropy':'ua'})
        df['beijing'] = df1
        
        df1= pd.read_csv('../data/usa/usa_cc.csv')
        df1 = df1.rename(columns={'HPI':'hpi','geo':'geo_centrality','deltap':'growth','geoid':'district_id'})
        df['usa'] = df1
        self.df =df
        
        

    def compare_regressions(self):
        y = 'growth'
        cols = ['ua','popdse','hpi','geo_centrality']
        df_indexes = ['istanbul','beijing','usa']
        combinations = [(0,1,2,3)]*3
# =============================================================================
#         for c  in df_indexes:
#             self.df[c]= normalise_data(self.df[c][cols+[y,'district_id']],omit=['district_id'])
#         self.run_regressions(df_indexes,cols,combinations,y)
# =============================================================================
        
        for city in df_indexes:
            df_idx = [city]*4
            combinations = [(0,),(0,1),(0,1,2),(0,1,2,3)]
            reg_order = cols + ['Intercept','Observations']
            model_names = ['Model 1','Model 2','Model 3','Model 4']
            self.run_regressions(df_idx,cols,combinations,y,model_names,reg_order,'control_{}'.format(city))
# =============================================================================
#         df_flow = read_in_out_diversity(is_repeat=True)
#         df_flow = normalise_data(df_flow)
#         self.df['istanbul'] = self.df['istanbul'].join(df_flow,on='district_id',how='left') 
#         
#         extra_cols= ['composite_demographic_inflow_entropy',
#        'composite_demographic_inflow_num_unique',
#        'composite_demographic_inflow_total']
#         
#         for extra_col in extra_cols:
#             cols = [extra_col,'ua','popdse','hpi','geo_centrality']
#             df_indexes = ['istanbul']*3
#             combinations = [(0,),(0,2,3,4),(0,1,2,3,4)]
#             self.run_regressions(df_indexes,cols,combinations,y,'extra_{}'.format(extra_col))
#     
# =============================================================================
    def run_for_results(self):
        self.compare_regressions()
  

def read_usa_data():
    df = pd.read_csv('../data/usa/usa_cc.csv')
    df.rename(columns = {'HPI':'hpi','geoid':'district_id','popu':'population'},inplace=True)
    df.drop(columns=['area','popdse'],inplace=True)
    df['deltaperc'] = df.deltap*100
    df_area = pd.read_csv('../data/usa/CensusData/area.csv').rename(columns={'GEOID':'district_id','Areaname':'district_name'})
    df = pd.merge(df,df_area,on='district_id',how='left')
    df['area_km2'] = df.area*2.58999
    df['popdse'] = df.population/df.area_km2
    return df

def read_yelp_shop_data():
    shops=json.load(open('../data/usa/yelpData/yelpShop.json','rb'))
    df = pd.DataFrame.from_dict(shops,orient='index')
    df.rename(columns = {'bizID': 'shop_id','categories':'category'},inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df
def read_yelp_consumption_data():
    consume = json.load(open('../data/usa/yelpData/yelpshop-consumption.json','rb'))
    dfs={}
    for year in [2011,2012,2013,2014,2015]:
        df =  pd.DataFrame.from_dict(consume[str(year)],orient='index')
        df['shop_id'] = df.index
        df.rename(columns={0:'consumption'},inplace=True)
        df.reset_index(drop=True,inplace=True)
        dfs[year] = df
        
    return dfs

def combine_yelp_data(df,shop):
    df = pd.merge(df,shop[['shop_id','category','lat','lon']],on='shop_id',how='left')
    df.dropna(subset=['category'],inplace=True)
    return df

        
class SupplementaryUSA(Result):
    name = 'supplementary_usa'
    
    def __init__(self):
        self.df = read_usa_data()
        shop = read_yelp_shop_data()
        dfs = read_yelp_consumption_data()
        for y in [2011,2012,2013,2014,2015]:
            dfs[y] = combine_yelp_data(dfs[y],shop)
        self.dfs = dfs
        
    
    def plot_maps(self):
        df = self.df
        
        self.plot_map(df, 'deltaperc', 'Economic Growth (%)', 'econ_growth')
        self.plot_map(df,'population','Population (in thousands)','population')
        
        
    def plot_bars(self):
        df= self.df
        df.sort_values(by='population',inplace=True)
        self.plot_bar(df, 'population', ['' for _ in range(len(df))],'Population (in thousands)','population')
        self.plot_bar(df, 'popdse', ['' for _ in range(len(df))],'Density (in thousands/km2)','population_density')
        self.plot_bar(df,'area_km2',['' for _ in range(len(df))],'Area (in km2)','area')
        
    def plot_shop_categories(self):
        df = pd.DataFrame()
        for y in [2011,2012,2013,2014,2015]:
            df = df.append(self.dfs[y])
        cat_bar = pd.DataFrame(df.category.value_counts().rename('count'))
        
        cat_bar.index = [c for c in cat_bar.index]
        
        cat_bar = cat_bar.iloc[0:40]
        self.plot_bar(cat_bar,'count',cat_bar.index.values,'Number of Transactions per Category','usa_trans_cat_count')
    
    def plot_trans_summary(self):
        df = pd.DataFrame()
        districts = self.df.district_id.astype(str).unique()
        for y in [2011,2012,2013,2014,2015]:
            df = df.append(self.dfs[y])
        
        def get_trans_per_shop(chunk):
            return pd.Series({'lat':chunk.lat.iloc[0],'lon':chunk.lon.iloc[0],'count':1})
        trans_count = df.groupby('shop_id').apply(get_trans_per_shop)
        trans_count.dropna(subset=['lat','lon'],inplace=True)
        scale = 1
        self.plot_bubble_map(trans_count, 'count', scale,districts, 'trans_count')
        
    def run_for_results(self):
# =============================================================================
#         self.plot_maps()
#         self.plot_bars()
#         self.plot_shop_categories()
#         self.plot_trans_summary()
# =============================================================================
        self.get_summaries()
        
    def get_summaries(self):
        df = self.df
        for col in ['population','area_km2','popdse']:
            print('======{}====='.format(col))
            print ('Min: {}'.format(df[col].min()))
            print ('Mean: {}'.format(df[col].mean()))
            print ('Median: {}'.format(df[col].median()))
            print ('Max: {}'.format(df[col].max()))
            print('\n\n')

results=[]

results.append(Supplementary())
# =============================================================================
# results.append(HuffModel())
# results.append(CommoditiesAndFlow())
# =============================================================================
# =============================================================================
# results.append(FlowAndEconomicOutput())
# =============================================================================
# =============================================================================
# results.append(SupplementaryChina())
# =============================================================================
# =============================================================================
# # =============================================================================
# =============================================================================
# results.append(GrowthVsDiv())
# =============================================================================
# =============================================================================
# results.append(SupplementaryUSA())
# =============================================================================

for result in results:
    result.run_for_results()


    

