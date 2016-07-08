import pandas as pd
import numpy as np

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

#fill in mising data
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

#choose country DE
df2 = df[df.country=='de']
df3 = df[df.channel=='Amazon']
df4 = df[df.refined_click > 5000]

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

import matplotlib.pyplot as plt

amazonClicks = groupedDf[groupedDf.index.get_level_values('channel')=='Amazon'].refined_click.values

amazonClicksSeries = pd.Series(amazonClicks, index=groupedDf.index.get_level_values('date').unique())
plt.figure
amazonClicksSeries.plot()
plt.show(block=False) # show argument block is used to enable continuing with the code


#create data frame for refined clicks for all channels: use debug to show how the data frame is built

refinedClicksDf = pd.DataFrame(index=groupedDf.index.get_level_values('date').unique())
for c in channels:
    channelDF = pd.DataFrame(df3[df3.channel == c])
    channelSeries = channelDF.groupby(['date']).sum()
    refind = pd.DataFrame(channelSeries.refined_click).rename(columns={'refined_click': c + '_refined_clicks'})
    refinedClicksDf = pd.concat([refinedClicksDf, refind], join='outer', axis=1)



