import pandas as pd
import numpy as np

#########################################
#create data frame
########################################
# from series
s = pd.Series([1,3,5,np.nan,6,8])

#create data frame, pass dates argument as index
dates =  pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

#from a dictionary of Series
dict = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
        'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
pd.DataFrame(dict, index=['d', 'b', 'a'])

#from a dictionary of lists
dict = {'one' : [1., 2., 3., 4.],'two' : [4., 3., 2., 1.]}
pd.DataFrame(dict, index=['a', 'b', 'c', 'd'])

#from a list of dictionaries
dict = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
pd.DataFrame(dict)

#create a multi-indexed frame by passing a tuples dictionary
pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
              ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
              ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
              ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}})

############################################
#read in data frame
############################################
root= '/media/sf_Haddie/Documents/data_example.csv'
df = pd.read_csv(root)
df.shape
# we notive that df has one column
df.head()
#separation is by semicolon
df = pd.read_csv(root, sep=';')
df.shape
df.head()
#we see that we have some NANs, we want to know how many missing data do we have
sum(pd.isnull(df.values))#we get an array with the sum of missing values in each column
sum(pd.isnull(df.values.ravel())) #we get the sum of all missing values

#############################################
#fill in mising data
#############################################
df2 = df
df2 = df2.fillna(method='backfill') #use next valid observation
sum(pd.isnull(df2.values))

df3 = df2.fillna(method='ffill') #use last valid observation
sum(pd.isnull(df3.values))

df2 = df.fillna(0) #fill in with zeros
sum(pd.isnull(df2.values))

#fill with the mean of each column
df.mean()
df2 = df.fillna(df.mean()) #fill in with mean columns
sum(pd.isnull(df2.values)) # columns 1:3 are strings. hence there is no replacement

# using a function to fill in with mean columns. we exclude the string columns
n = len(df.columns)
df2 = df.iloc[:,4:n].apply(lambda x: x.fillna(x.mean()),axis=0)
sum(pd.isnull(df2.values))

df = df.fillna(0)
sum(pd.isnull(df.values))
df.iloc[:,1:3] = df.iloc[:,1:3].apply(lambda x: x.replace(0,'no info'))


#############################
#select by label
#############################
df['refined_click'][1:10]
#a list of labels
df[['channel','refined_click']][1:10]
df[1:10][['channel','refined_click']]
#or
vars = ['channel', 'activity', 'refined_click']
df[vars][1:10]

#select a range of labels using .loc
df.loc[1:10,'channel':'refined_click']
#select specific rows
df.loc[[1,2,5,7],'channel']

df.loc['channel'] #with loc at least one selection has to be an index

#create df5 with an index of dates
df5 = pd.DataFrame(df, index=df.date)
df6 = df5.loc['02.04.2015','channel'] #now it works because the date is an index
df6 = df5.loc['02.04.2015':'05.04.2015','channel'] #index range
df6.index.unique()

######################################
#select by position
######################################
#using iloc : a purely integer based indexing. start bound is included, upper bound is exluded

df7 = df[vars][1:10]
df7.shape

df7.iloc[0:3]
df7.iloc[1:2,:]
df7.iloc[:,1:2]
df7.iloc['activity'] #doesn't work : integer based index
# select specific rows and columns
df7.iloc[[1,5,7],1:3]
df7.iloc[[1,5,7],2]
df7.iloc[[1,5,7]]
df7.iloc[[1,5,7],[1,2]]

#change some values in specific positions
df7.iloc[[1,5,7],2]=7
df7.iloc[[1,5,7],2]

#out of bound selection: works only in Pandas version above 0.14.0
df7.iloc[2:15,1:2]
df7.iloc[2:15,1:7]
df7.iloc[2:15,3:7] #returns empty data frame
df7.iloc[:,5]   #returns an IndexError because it is a single index


####################################
#conditional selection and callable
####################################
df2 = df[df.country=='de'] #country de
df3 = df[df.channel=='Amazon'] #channel amazon
df4 = df[df.refined_click > 5000] # clicks bigger than 5000
#select rows
df[0:10] # select first 10 rows
df[0:10][1:5] #select from them 4 rows starting from 2 position

#callable
df7.loc[lambda df7: df7.refined_click==7, :]
df7.loc[lambda df7: df7.channel=='Affiliate', :]
df7.loc[lambda df7: df7.channel!='Affiliate', :] #not equal to



#how many channels do we have?
channels = df2.channel.unique()
len(channels) #number of different channels

#for each channels, how many activities do we have
act = dict()
for c in channels:
   act[c] =len(df2[df.channel==c].activity.unique())

actNames = dict()
for c in channels:
    actNames[c] =tuple(df2[df.channel==c].activity.unique())



#group data by date and channel
df3 = df[df.country=='de']
df3 = df3.drop('activity',1)

groupedDf = df3.groupby(['date','channel']).sum()
################################
#view data
################################
#pip install matplotlib
import matplotlib.pyplot as plt

amazonClicks = groupedDf[groupedDf.index.get_level_values('channel')=='Amazon'].refined_click.values

amazonClicksSeries = pd.Series(amazonClicks, index=groupedDf.index.get_level_values('date').unique())
plt.figure
amazonClicksSeries.plot()
plt.show(block=False) # show argument block is used to enable continuing with the code
plt.savefig(root+'plt.svg')

##############################################
#joining data
##############################################
#simple examples to illustrate the different types of joining data frames
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']})

merged = pd.merge(left,right,on='key')

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

merged = pd.merge(left,right,on=['key1','key2'])

#inner join: use intersection keys to combine data frames in a new data frame that contains only rows that have matching values in both of the original data frames
innerMerged = pd.merge(left, right, how='inner', on=['key1','key2'])

#left join: use keys from left table only to combine data frames in a new data frame that will contain all rows from the left data frame even if they don't have matching values in the right data frame
leftMerged = pd.merge(left, right, how='left', on=['key1','key2'])

# right join: opposite to left join
rightMerged = pd.merge(left, right, how='right', on=['key1','key2'])

#outer join: use union of keys in both data frames to combine data frames
outerMerged = pd.merge(left, right, how='outer', on=['key1','key2'])


#joining on indices: data alignment is on the indices

left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']},
                     index=['K0', 'K1', 'K2', 'K3'])
right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                      index=['K0', 'K1', 'K3', 'K4'])

merged = left.join(right) # this is a left join. The function uses the keys from the calling data

outerMerged = left.join(right, how='outer')
outerMerged = pd.merge(left, right, left_index=True, right_index=True, how='outer')

innerMerged = left.join(right, how='inner')
innerMerged = pd.merge(left, right, left_index=True, right_index=True, how='inner')

left = pd.DataFrame({'key': ['K0', 'K1', 'K1', 'K0'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'C': ['C0', 'C1'],
                      'D': ['D0', 'D1']},
                     index=['K0', 'K1'])

merged = pd.merge(left,right, left_on='key',right_index=True, how='left', sort=False)


#create data frame for refined clicks for all channels: use debug to show how the data frame is built

refinedClicksDf = pd.DataFrame(index=groupedDf.index.get_level_values('date').unique())
for c in channels:
    channelDF = pd.DataFrame(df3[df3.channel == c])
    channelSeries = channelDF.groupby(['date']).sum()
    refind = pd.DataFrame(channelSeries.refined_click).rename(columns={'refined_click': c + '_refined_clicks'})
    refinedClicksDf = pd.concat([refinedClicksDf, refind], join='outer', axis=1)

#some statistical insights
#correlation
stat = pd.stats.moments

refinedClicksDf['Amazon_refined_clicks'].corr(refinedClicksDf['SEO_refined_clicks'])
movingAve = stat.rolling_mean(refinedClicksDf['Amazon_refined_clicks'],10)
plt.figure
movingAve.plot()
plt.show(block=False)
#regression analysis
#install statsmodels : pip install statsmodels (might ask for more packages to install: e.g. scipy, patsy...)
import statsmodels.formula.api as smf
model = smf.OLS(refinedClicksDf['Amazon_refined_clicks'], refinedClicksDf['SEO_refined_clicks'])
res = model.fit()
print res.summary()


#test residuals normality
import statsmodels.stats.api as ss
ss.jarque_bera(res.resid)