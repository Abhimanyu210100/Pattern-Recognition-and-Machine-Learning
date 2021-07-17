#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:50:26 2021

@author: Abhimanyu
"""


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

bikers = pd.read_csv('../../bikers.csv')
train = pd.read_csv('../../train.csv')
test = pd.read_csv('../../test.csv')
bikers_network = pd.read_csv('../../bikers_network.csv')
tours = pd.read_csv('../../tours.csv')
tour_convoy = pd.read_csv('../../tour_convoy.csv')


stours = tours.rename({'biker_id':'organizer'},axis=1)


#Extracting date info from test and train data, year ignored because its the same everywhere
train.timestamp = pd.to_datetime(train.timestamp)
train['join_day'] = pd.DatetimeIndex(train.timestamp).day
train['join_month'] = pd.DatetimeIndex(train.timestamp).month

test.timestamp = pd.to_datetime(test.timestamp)
test['join_day'] = pd.DatetimeIndex(test.timestamp).day
test['join_month'] = pd.DatetimeIndex(test.timestamp).month

#Merging datasets
joint_data = pd.concat([train,test]).reset_index().drop(columns=['index'])
joint_data = pd.merge(joint_data,stours,on='tour_id',how='left')

#Extracting info from date (tours)
joint_data.tour_date = pd.to_datetime(joint_data.tour_date)
joint_data['tour_year'] = pd.DatetimeIndex(joint_data.tour_date).year
joint_data['tour_month'] = pd.DatetimeIndex(joint_data.tour_date).month
joint_data['tour_day'] = pd.DatetimeIndex(joint_data.tour_date).day


#Extracting friends features
joint_data['organizer_is_biker'] = 0
joint_data['organizer_is_friend'] = 0
joint_data['friends_going'] = 0
joint_data['friends_not_going'] = 0
joint_data['friends_maybe'] = 0
joint_data['friends_invited'] = 0

for i in tqdm(range(len(joint_data))):
    b = joint_data.biker_id.iloc[i]
    t = joint_data.tour_id.iloc[i]
    org = joint_data.organizer.iloc[i]
    
    ind1 = int(np.array(np.where(bikers_network.biker_id == b)))
    temp_network = bikers_network.iloc[ind1,:]
    
    ind2 = int(np.array(np.where(tour_convoy.tour_id == t)))
    temp_convoy = tour_convoy.iloc[ind2,:]
    
    count_going=0
    count_not_going = 0
    count_maybe = 0
    count_invited = 0
    
    if b == org:
        joint_data['organizer_is_biker'].iloc[i] = 1
    
    if str(temp_network.friends) != 'nan':
        list_frnds = temp_network.friends.split(' ')
        
        
        
        if str(temp_convoy.going) != 'nan':
            going = temp_convoy.going.split(' ')
            
            for j in range(len(list_frnds)):
                b_temp = list_frnds[j]
                if b_temp in going:
                    count_going+=1
        joint_data['friends_going'].iloc[i] = count_going
        
        
        if str(temp_convoy.not_going) != 'nan':
            not_going = temp_convoy.not_going.split(' ')
            
            for j in range(len(list_frnds)):
                b_temp = list_frnds[j]
                if b_temp in not_going:
                    count_not_going+=1
        joint_data['friends_not_going'].iloc[i] = count_not_going
        
        
        if str(temp_convoy.maybe) != 'nan':
            maybe = temp_convoy.maybe.split(' ')
            
            for j in range(len(list_frnds)):
                b_temp = list_frnds[j]
                if b_temp in maybe:
                    count_maybe+=1
        joint_data['friends_maybe'].iloc[i] = count_maybe
        
        
        if str(temp_convoy.invited) != 'nan':
            invited = temp_convoy.invited.split(' ')
            
            for j in range(len(list_frnds)):
                b_temp = list_frnds[j]
                if b_temp in invited:
                    count_invited+=1
        joint_data['friends_invited'].iloc[i] = count_invited           
                    
                    
                    
                    
        if org in list_frnds:
            joint_data['organizer_is_friend'].iloc[i] = 1
    

#Difference in time
joint_data['hours_bw_tour_join'] = ((joint_data.tour_date - joint_data.timestamp)/dt.timedelta(hours=1))

joint_data.drop(columns=['timestamp','tour_date','organizer'],inplace=True)

joint_data1 = pd.concat([train,test]).reset_index().drop(columns=['index'])
joint_data1 = pd.merge(joint_data1,stours,on='tour_id',how='left')


joint_data['timestamp'] = joint_data1.timestamp
joint_data = joint_data.rename({'hours_bw_tour_join':'hours_bw_inform_tour',
                               'join_day':'invite_day',
                               'join_month':'invite_month'},axis=1)

#Bikers date info
bikers = bikers[bikers['member_since'] != '--None']
bikers.member_since = pd.to_datetime(bikers.member_since)

bikers['join_year'] = pd.DatetimeIndex(bikers.member_since).year
bikers['join_month'] = pd.DatetimeIndex(bikers.member_since).month
bikers['join_day'] = pd.DatetimeIndex(bikers.member_since).day
bikers

#Difference in time
temp = joint_data
merged_data = pd.merge(temp,bikers,on='biker_id',how='left')
merged_data['hours_bw_invite_member'] = (merged_data.timestamp - merged_data.member_since)/dt.timedelta(hours=1)
merged_data['tour_date'] = joint_data1.tour_date
merged_data.tour_date = pd.to_datetime(merged_data.tour_date)
merged_data['hours_bw_tour_member']=(merged_data.tour_date-merged_data.member_since)/dt.timedelta(hours=1)
merged_data.drop(columns=['tour_date','member_since','timestamp'],inplace=True)

#Final data file
data = merged_data

#Creating train and test data
x_train = data.iloc[:len(train),:]
x_test = data.iloc[len(train):,:]
y_train = data.like.iloc[:len(x_train)]

x_train.drop(columns=['like','dislike','biker_id','tour_id'],inplace=True)
x_test.drop(columns=['like','dislike','biker_id','tour_id'],inplace=True)

#Changing the datatype of categorical features
cat_list = ['city','state','country','language_id','location_id','gender','area','pincode','bornIn']

for c in cat_list:
    x_train.loc[:,c] = x_train.loc[:,c].astype('category')
    x_test.loc[:,c] = x_test.loc[:,c].astype('category')

#Ranking
lgbm = LGBMClassifier()
lgbm.fit(x_train,y_train,feature_name=list(x_train.columns), categorical_feature=cat_list)

probabs1 = lgbm.predict_proba(x_test)[:,1]


z = pd.read_csv('../../test.csv')
z['probab'] = -probabs1
z.drop(columns = ['invited','timestamp'],inplace=True)
z.columns = ['biker_id','tour_id','probab']


sub = pd.read_csv('../../sample_submission.csv')
for i in range(len(sub)):
    b = sub.biker_id.iloc[i]
    
    temp = z[z.biker_id==b]
    order = temp.probab.argsort()
    sorted_list = list(temp.tour_id.iloc[order])
    sorted_list = " ".join(sorted_list)
    sub['tour_id'][i] = sorted_list
    sub['biker_id'][i] = b

sub.to_csv('MM17B001_MM17B008_1.csv',index=False)


#Catboost
x_train = data.iloc[:len(train),:]
x_test = data.iloc[len(train):,:]
y_train = data.like.iloc[:len(x_train)]

x_train.drop(columns=['like','dislike','biker_id','tour_id'],inplace=True)
x_test.drop(columns=['like','dislike','biker_id','tour_id'],inplace=True)
#Replacing Nan by median and none by mode
for c in x_train.columns:
    try:
        x_train.loc[:,c] = x_train.loc[:,c].fillna(x_train.loc[:,c].median().iloc[0])
        x_test.loc[:,c] = x_train.loc[:,c].fillna(x_test.loc[:,c].median().iloc[0])
    except:
        x_train.loc[:,c] = x_train.loc[:,c].fillna(x_train.loc[:,c].mode().iloc[0])
        x_test.loc[:,c] = x_test.loc[:,c].fillna(x_test.loc[:,c].mode().iloc[0])

#classifier
cat = CatBoostClassifier(allow_writing_files=False)
cat.fit(x_train,y_train, cat_features=cat_list)


probabs2 = cat.predict_proba(x_test)[:,1]
probabs = (probabs1+probabs2)/2



z = pd.read_csv('../../test.csv')
z['probab'] = -probabs
z.drop(columns = ['invited','timestamp'],inplace=True)
z.columns = ['biker_id','tour_id','probab']


sub = pd.read_csv('../../sample_submission.csv')
for i in range(len(sub)):
    b = sub.biker_id.iloc[i]
    
    temp = z[z.biker_id==b]
    order = temp.probab.argsort()
    sorted_list = list(temp.tour_id.iloc[order])
    sorted_list = " ".join(sorted_list)
    sub['tour_id'][i] = sorted_list
    sub['biker_id'][i] = b

sub.to_csv('MM17B001_MM17B008_2.csv',index=False)


























