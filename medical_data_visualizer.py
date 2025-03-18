import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv', index_col='id')

# 2
overweight = pd.Series(data= df['weight'] / ((df['height']/100)**2), name='overweight')
overweight.loc[overweight <= 25.0] = 0
overweight.loc[overweight > 25.0] = 1
df = df.join(overweight)

# 3
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    card_1 = df_cat.loc[df_cat['cardio'] == 1].value_counts()
    card_0 = df_cat.loc[df_cat['cardio'] == 0].value_counts()
    df_card0 = pd.DataFrame(card_0).reset_index().sort_values(by='variable', axis=0)
    df_card1 = pd.DataFrame(card_1).reset_index().sort_values(by='variable', axis=0)
    frames = [df_card0, df_card1]
    df_cat = pd.concat(frames)
    df_cat = df_cat.rename(columns={'count': 'total'})
    # 7
    df_cat['value'] = df_cat['value'].apply(lambda x: int(x))


    # 8
    #another way to make the figure as I thought it was what was bugging with the test_units
    #fig = sns.FacetGrid(df_cat, col="cardio", height=4, aspect=1.3)
    #fig.map_dataframe(sns.barplot, x="variable", y="total", hue="value", palette='deep')
    #fig.set_axis_labels('variable', 'total').set_xticklabels(labels=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
    #fig.add_legend(title='value')   
    fig = sns.catplot(data=df_cat, x="variable", y="total", col="cardio", kind="bar", hue="value")
    fig.set_ylabels('total')
    
    # 9
    fig=fig.fig
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    un_d_index = df[(df['ap_lo'] > df['ap_hi']) | (df['height'] < df['height'].quantile(0.025)) | (df['height'] > df['height'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025)) | (df['weight'] > df['weight'].quantile(0.975))].index
    df_heat = df.drop(un_d_index)
    df_heat.reset_index(inplace=True)

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.tri(corr.shape[0], corr.shape[1], k=0).T
    #the comented line bellow is to get the 'sex' column label as 'gender' (as in the example 2 picture) though the tests check for the 'sex' label value
    #df_heat.rename(columns={'sex': 'gender'}, inplace=True)



    # 14
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(11,9))

    # 15
    sns.heatmap(corr, vmin=-0.16, vmax=0.32, center=0, annot=True, fmt='.1f', linewidth=.5, cbar_kws={'shrink': 0.5, 'ticks':[-0.08, 0, 0.08, 0.16, 0.24]}, mask=mask, ax=ax)


    # 16
    fig.savefig('heatmap.png')
    return fig
