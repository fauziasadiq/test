import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')
#WHAT IS IN THIS CODE?
'''
This is the sample sales data for 4 products in Ontario and Alberta.
Each region has one banner and each banner has multiple stores
Find top selling product in Ontario and predict unit volume for last 12 weeks
on basis of pricing and ad information. 
'''

def add_plot_cosmetics(x,y,ptitle):
    plt.title(ptitle)
    plt.xticks(rotation=90)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.grid()
    plt.show()
    
df = pd.read_csv('salesdata2018.csv')
# Change data type
df['YEAR WEEK']=df['YEAR WEEK'].astype(str)
#Get week number and make it integer
df['WEEK']=df['YEAR WEEK'].str[4:]
df['WEEK']=df['WEEK'].astype(int)
#Find if a week is Ad week or not (1/0 means yes/no)
df['IsAD'] = np.where(df['FLYER PAGE'].isnull(),0,1)
#How many stores are in each regions
regions = df['REGION'].unique()
for reg in regions:
    nstores = df['STORE CODE'][df['REGION']==reg].nunique()
    print(f"{reg} has {nstores} stores")
# Banner/Region-wise weekly product sales
agg_dict = {'UNITS':'sum','BASE':'sum','DOLLARS':'sum','SOLD PRICE':'mean',
            'REGULAR PRICE':'mean', 'IsAD':'mean'}
agg_cols = ['YEAR WEEK','PROD CODE','BANNER', 'REGION']
dp       = df.groupby(agg_cols).agg(agg_dict).reset_index()

# All year product-wise dollars in ONTARIO
rev = dp.groupby(['PROD CODE','REGION'])['DOLLARS'].sum().reset_index()
x, y = 'PROD CODE', 'DOLLARS'
plt.figure(figsize=(10,5))
reg='ONTARIO'
xx = rev[x][rev['REGION']==reg].astype(str)
yy = rev[y][rev['REGION']==reg]
plt.bar(xx,yy,alpha=0.7, label=reg)
ptitle='Product Revenue'
add_plot_cosmetics(x,y,ptitle)

#Choose biggest rev product in ontario
max_rev = rev['DOLLARS'][rev['REGION']==reg].max()
top_prod = rev['PROD CODE'][rev['DOLLARS']==max_rev].iloc[0]
dtop = dp[(dp['REGION']==reg) & (dp['PROD CODE']==top_prod)]
bnr  = dtop['BANNER'].iloc[0]
# Total sales volume and base sales volume (without promotion)
x, y, y1,y2 = 'YEAR WEEK', 'UNITS', 'BASE','IsAD'
xx, yy, yy1,yy2 = dtop[x], dtop[y], dtop[y1], dtop[y2]
max_vol = dtop[y].max()
plt.figure(figsize=(12,6))
plt.bar(xx,yy,alpha=0.5,label=y)
plt.bar(xx,yy1,alpha=0.5,label=y1)
plt.scatter(xx,yy2*max_vol/2,label=y2,color='red',s=100,marker='*')
ptitle="Product/BANNER/REGION : "+str(top_prod)+" / "+bnr+" / "+reg
add_plot_cosmetics(x,y,ptitle)

# LINEAR REGRESION TO PREDICT LAST 12 WEEKS of UNIT SALES VOLUME
dr = dtop.copy()
dr['YEAR WEEK'] = dr[x].astype(int)
# Split data into train and test
train, test = dr[dr[x]<=201840],dr[dr[x]>=201841]
x_var    = ['SOLD PRICE', 'REGULAR PRICE', 'IsAD']
x_train  = train[x_var].values.reshape(-1,len(x_var))
y_train  = train[y]
x_test   = test[x_var].values.reshape(-1,len(x_var))
y_act    = test[y].tolist()
ols      = linear_model.LinearRegression()
model    = ols.fit(x_train, y_train)
test['UNITS PRD']   = model.predict(x_test)
test[x] = test[x].astype(str)
plt.figure(figsize=(12,6))
plt.plot(test[x],test[y], label='ACTUAL UNITS')
plt.plot(test[x],test['UNITS PRD'], label='PREDICTED UNITS')
ptitle="Prediction of last 12 weeks 2018 for Product "+str(top_prod)
add_plot_cosmetics(x,y,ptitle)