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


import folium

import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

sns.set(color_codes=True)
sns.set_style("whitegrid")

EXCLUDE = [1,131,111]

def normalise_data(df,omit=[]): 
    df2 = df.copy(deep=True)
    cols=df.columns.tolist()
    for o in omit:
        cols.remove(o)
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col]-mean)/std
    return df2

#helper to read raw credit card data , repeat means we consider more than one trip by a user
def read_raw_flows(is_repeat,omit=True):
    print("Reading transcation data...\n")
    df_trans = pd.read_csv('../data/istanbul/0-transactions.txt')
    
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
    flowmat = get_flow_matrix(is_repeat)
    within_flow = flowmat[flowmat.hdistrict_id == flowmat.sdistrict_id].drop(columns=['sdistrict_id'])
    within_flow.rename(columns= {'flow': 'withinflow','hdistrict_id':'district_id'},inplace=True)
    within_flow.index=within_flow.district_id
    within_flow.drop(columns =['district_id'],inplace=True)
    outflow = flowmat.groupby('hdistrict_id')['flow'].sum().rename('outflow')
    outflow.index.rename('district_id',inplace=True)
    
    inflow = flowmat.groupby('sdistrict_id')['flow'].sum().rename('inflow')
    inflow.index.rename('district_id',inplace=True)
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
def read_in_out_diversity():
    demog = get_customer_info()
    demog.index = demog.customer_id
    demog.drop(columns=['customer_id','hdistrict_id'],inplace=True)
    df = read_raw_flows(is_repeat=True)
    df = df.join(demog,on='customer_id',how='left') 
    assert not df.incomecat.isnull().sum()
    #df.dropna(subset=['hdistrict_id'],inplace=True)
    #'vector' that describes the diversity of the people
#==============================================================================
#     df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) + df.gender.astype(str)
#==============================================================================
    df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.wdistrict_id.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) + df.gender.astype(str)
#==============================================================================
#      df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) + df.gender.astype(str)
#==============================================================================
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
    
    def write_regression_result(self,res,f_name):
        result_location = '../results/{}/regressions/indv_results'.format(self.name)
        create_dir(result_location)
        with open(os.path.join(result_location,'{}.tex'.format(f_name)), 'w') as f:
            f.write(res.summary().as_latex())
        print(res.summary())
    
    def write_regressions_summary(self,reses,y):
        info_dict = {'N':lambda x: "{0:d}".format(int(x.nobs)),
                     'R2':lambda x: "{:.2f}".format(x.rsquared),
                     'Adjusted R2':lambda x: "{:.2f}".format(x.rsquared_adj)}       
        dfoutput = summary_col(reses,stars=True,info_dict=info_dict)
        result_location = '../results/{}/regressions/summary'.format(self.name)
        create_dir(result_location)
        self.reg_table = dfoutput
        with open(os.path.join(result_location,'{}.tex'.format('{}_regression_summary_table'.format(y))), 'w') as f:
            f.write(dfoutput.as_latex())
    
    def plot_bar(self,df,col,ticks,y_lab,name_suffix):
        colors = ['#EF476F','#00CFC1','#086788','#FF9F1C','#EB5E28']
        color = colors[df.columns.get_loc(col)%len(colors)]
        
        
        y_pos = np.arange(len(ticks))
        values = df[col].values
        
        plt.figure()
        plt.bar(y_pos, values, align='center', alpha=0.5,color=color)
        plt.xticks(y_pos, ticks, rotation=270)
        plt.ylabel(y_lab)

        plt.tight_layout()
        self.save_plot('bar_{}.png'.format(name_suffix))
    
    def plot_heatmap(self,matrix,cmap,tick_label_x,tick_label_y,bar_label,name_suffix):
        plt.figure()
        sns.heatmap(matrix,cmap=cmap,cbar_kws={'label': bar_label})
        plt.yticks(np.arange(len(tick_label_y)),tick_label_y,size=8)
        plt.xticks(np.arange(len(tick_label_y)),tick_label_x,rotation=270,size=8,verticalalignment='top')
        plt.tight_layout()
        self.save_plot('heatmap_.png'.format(name_suffix))
        
    def run_regressions(self,df_indexes,cols,combinations,y):
        reses = []
        assert len(df_indexes) == len(combinations)
        for i in range(len(combinations)):
            df = self.df[df_indexes[i]]
            combination = combinations[i]
            xs = [cols[c] for c in combination]
            reses.append(self.regress(df,xs,y))
        self.write_regressions_summary(reses,y)
            
    def regress(self,df,xs,y,f_name=None):

        x_string='+'.join(xs)
        mod = smf.ols('{} ~ {}'.format(y,x_string), data=df)
        res = mod.fit()

        if not f_name:
            f_name = '{}_{}'.format(y,x_string) 
        self.write_regression_result(res,f_name)
        
        self.plot_residuals(df[y].values,res.fittedvalues,y,f_name='resid_{}'.format(f_name))
        return res
    
    def plot_residuals(self,y,y_fitted,y_name,f_name):
        df=pd.DataFrame()
        
        df.loc[:,'errors'] =  y - y_fitted
        df.loc[:,'y'] = y
        df.loc[:,'y_fitted'] = y_fitted
        self.plot_scatter(df,'y_fitted','errors',x_lab ='Fitted {}'.format(y_name),y_lab='residuals',f_name=f_name)
        
                
    def save_plot(self,fig_name):
        plot_location = '../results/{}/plots'.format(self.name)
        create_dir(plot_location)
        plt.savefig(os.path.join(plot_location,'{}'.format(fig_name)),dpi=300)
    
    def get_corr(self,df,x,y):
        return df[[x, y]].corr()[y][x]
    
    def plot_scatter(self,df,x,y,x_lab=None,y_lab=None,has_best_fit=False,controls=None,f_name=None):
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
            plt.text(df[x].mean(),df[y].mean(),'Correlation : {}'.format(corr),size=15)
            plt.plot(np.unique(df[x].values), np.poly1d(np.polyfit(df[x].values, df[y].values, 1))(np.unique(df[x].values)))
        else:
            corr=0
        plt.tight_layout()
        self.save_plot('scatter_{}'.format(f_name))
        
        return corr
    def plot_map(self,df,col,legend_string,name):
        state_geo = '../data/istanbul/district_level_shape/district.geojson'
    #==============================================================================
    #     state_geo = 'td/bj_shapefile/bj_shapefile.geojson'
    #==============================================================================
        # Initialize the map:
        locs = [[41.0082, 28.9784]]
        m = folium.Map(location=locs[0], zoom_start=10,tiles='cartodbpositron')
         
        # Add the color for the chloropleth:
        style_function = lambda x: {'fillOpacity': 1 if       
                                x['properties']['districtid']==0 else
                                 0, 'fillColor' : "#000000",'stroke': False }
    
        mapbg=folium.GeoJson(state_geo,style_function)
        
        m.choropleth(
         geo_data=state_geo,
         name='choropleth',
         data=df,
         columns=['district_id',col],
         key_on='feature.properties.districtid',
         fill_color='PuBu',
         fill_opacity=0.7,
         line_opacity=0.2,
         legend_name=legend_string
        )
        mapbg.add_to(m)
        # Save to html
        location = '../results/{}/maps'.format(self.name)
        create_dir(location)
        m.save(os.path.join(location,'map_{}.html'.format(name)))
        
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
        y_lab= "Economic Productivity"
        x_lab='Volume of Total Flow'
        x_lab = None
        y_lab = None
        xs = ['inflow','outflow','totalflow']
        ys = ['log_{}'.format(c) for c in ['y2014','y2015','y2016']]
    
        df = pd.DataFrame()
        for x in xs:
            for y in ys:
                corr=self.plot_scatter(self.df['istanbul'],x,y,x_lab,y_lab,has_best_fit=True)
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
            self.run_regressions(df_index,cols,combinations,y)
        
    def run_for_results(self):
        self.plot_results()
        self.get_regressions()


def read_areas():
    df = pd.read_csv('../data/istanbul/Istanbul_district_area.txt')
    df = df[~df.district_id.isin(EXCLUDE)]
    df.sort_values(by='district_id',inplace=True)
    return df
    
def read_poi():
    poi = pd.read_csv('../data/istanbul/attractiveness.csv')
    poi.rename(columns={'POLYGON_NM':'district_name'},inplace=True)
    to_remove = ['MajHwys','SecHwys','Parking','RailRds']
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


def get_total_flows():
    flow = get_flow_matrix(is_repeat=True)
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
        totalflow = get_total_flows()       
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
    
    def run_for_results(self):
        self.plot_poi_bars()
        self.plot_flow_heatmap()
        self.plot_distance_heatmap()
        self.run_ols()
        self.run_glms()
    
    def run_glm(self,y,xs,f_name=None):
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
        self.plot_residuals(df[y], res_nb.fittedvalues, 'Flow Counts', 'resid_nb_{}'.format(f_name))
        self.plot_residuals(df[y], res_poisson.fittedvalues, 'Flow Counts', 'resid_poisson_{}'.format(f_name))
        
        
        #### comparing mse
        reses = {'poisson':res_poisson, 'nb':res_nb}
        for res in reses:
            
            error = (df[y] - reses[res].fittedvalues) 
            mean_square_error = np.sqrt(error*error).mean()
            abs_error = abs(error).mean()
            print(mean_square_error,file=open('../results/{}/regressions/mse_{}.txt'.format(self.name,res),'w'))
            print(abs_error,file=open('../results/{}/regressions/mae_{}.txt'.format(self.name,res),'w'))
        return res_poisson,res_nb
    
            
    def run_glms(self):

        for n in range(self.REMOVE_RANGE):
            x_cols = ['poi_sum_{}_j'.format(n),'poi_diversity_{}_j'.format(n),'distance_ij']
            y = 'totalflow'
            xs = ['log_{}'.format(x) for x in x_cols]
            xs_control = ['log_{}'.format(x) for x  in ['distance_ij']]
            pois,nb = self.run_glm(y,xs,str(n))
            pois_control,nb_control = self.run_glm(y,xs_control,'control_{}'.format(n))
            poi_r2 = 1-(pois.deviance/pois.null_deviance)
            nb_r2= 1-(nb.deviance/nb.null_deviance)
            poi_r2_control = 1-(pois_control.deviance/pois_control.null_deviance)
            nb_r2_control = 1-(nb_control.deviance/nb_control.null_deviance)
            print ('proposed poi : {}'.format(poi_r2))
            print ('proposed nb : {}'.format(nb_r2))
            print ('====')
            print ('control poi : {}'.format(poi_r2_control))
            print ('control nb : {}'.format(nb_r2_control))
        
    def run_ols(self):
        
# =============================================================================
#         xs = ['poi_sum_j','poi_diversity_j','distance_ij']        
# =============================================================================
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
# =============================================================================
#             for i, chunk in df_norm.groupby('district_i'):
#     
#                 reg = self.regress(chunk,xs2,y,f_name='huff_ols_district_{}_{}'.format(i,n))
#                 df_params = df_params.append(pd.DataFrame(reg.params).transpose())
#                 districts.append('district_{}'.format(int(i)))
#                 regs.append(reg)
#             
#             df_params = df_params.append(pd.DataFrame(df_params.mean()).transpose())
#             districts.append('average')
# =============================================================================
            
            reg = self.regress(df_norm,xs2,y,f_name='huff_ols_global_{}'.format(n))
            regs_global.append(reg)
            regs.append(reg)
            df_params = df_params.append(pd.DataFrame(reg.params).transpose())
            districts.append('global')
            
            df_params['district_id'] = districts
            self.write_regressions_summary(regs,'huff_{}'.format(n))
            parameters_loc = '../results/{}/regressions/parameters'.format(self.name)
            create_dir(parameters_loc)
            df_params.to_csv(os.path.join(parameters_loc,'parameters_{}.csv').format(n))
            
  
            def normalize_prob(chunk):
                chunk = chunk.copy()
                norm = chunk.prob.sum()
                chunk.loc[:,'prob'] = chunk.prob/norm
                chunk.loc[:,'fitted_flow'] = chunk.prob * chunk.totalflow_not_norm.sum()
                return chunk
            df_norm.loc[:,'prob']=np.exp(reg.fittedvalues)
            df_norm = df_norm.groupby('district_i').apply(normalize_prob)
                
                
            
            error = (df_norm['totalflow'] - df_norm['fitted_flow']) 
            mean_square_error = np.sqrt(error*error).mean()
            abs_error = abs(error).mean()
            df_norm.to_csv('../results/{}/regressions/df_ols.csv'.format(self.name))
            print(mean_square_error,file=open('../results/{}/regressions/mse_{}.txt'.format(self.name,'ols'),'w'))
            print(abs_error,file=open('../results/{}/regressions/mae_{}.txt'.format(self.name,'ols'),'w'))
        
        self.write_regressions_summary(regs_global,'huff_globals'.format(n))
        
        
        
    def plot_poi_bars(self):
        cols = self.poi.columns.to_list()
        omit = ['district_name','district_id','poidensity','divcount']
    
        for o in omit:
            cols.remove(o)
        for col in cols:
            y_lab = 'Counts'    
            self.plot_bar(self.poi,col,self.poi.district_name,'{} {}'.format(col,y_lab),col)

        
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
    def get_population_plots(self):
        df = self.df['pop'].join(self.df['rep'],how='left')
        df['district_id'] = df.index
        self.plot_map(df,'population','Population (in thousands)','population')
        self.plot_map(df,'pop_rep','Sample counts','sample')
        self.plot_scatter(df,'pop_rep','population','Sample Size','Population Size',has_best_fit=True)
        
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
        
    def run_for_results(self):
        self.get_population_plots()
        self.get_area_plots()
        self.get_rent_plots()

results=[]

results.append(Supplementary())
results.append(HuffModel())
# =============================================================================
# results.append(CommoditiesAndFlow())
# =============================================================================
# =============================================================================
# results.append(FlowAndEconomicOutput())
# =============================================================================

for result in results:
    result.run_for_results()

#%%

    

#functions for exploratory purposes SI


def getHouseAndPopDseMap():
    df=pd.read_csv("thesisData/beijing_cc.csv")
    df2=pd.read_csv("yelp/istanbul_cc_n.csv")
    df3=pd.read_csv("thesisData/usa_cc.csv")
    x="popdse"
    y="HPI"
    df=df[df.HPI!=0]
    mapCreate(df2,"istan_HPI",y)
    mapCreate(df2,"istan_popdse",x)
#==============================================================================
#     plot(df3,x,y)
#==============================================================================
    
    



    
if __name__ == "__main__":
    getHouseAndPopDseMap()
    pass

