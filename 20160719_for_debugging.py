import pandas as pd

root= '/media/sf_Haddie/Documents/data_example.csv'
df = pd.read_csv(root, sep=';')
df.shape
df = df.fillna(0)
#group data by date and channel
df3 = df[df.country=='de']
df3 = df3.drop('activity',1)
groupedDf = df3.groupby(['date','channel']).sum()

channels = df3.channel.unique()

refinedClicksDf = pd.DataFrame(index=groupedDf.index.get_level_values('date').unique())
for c in channels:
        channelDF = pd.DataFrame(df3[df3.channel == c])
        channelSeries = channelDF.groupby(['date']).sum()
        refind = pd.DataFrame(channelSeries.refined_click).rename(columns={'refined_click': c + '_refined_clicks'})
        refinedClicksDf = pd.concat([refinedClicksDf, refind], join='outer', axis=1)

