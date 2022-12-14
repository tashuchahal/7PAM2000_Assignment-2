import pandas as pd                     # importing libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def get_data_frames(file,ind):  # Defining function
    """
    Returns
    -------
    df : TYPE
        DESCRIPTION.
    df2 : TYPE
        DESCRIPTION.

    """
    df = pd.read_csv(file,skiprows=(4),index_col=(False))  
    df = df.loc[:, ~df.columns.str.contains('Unnamed')] 
    
    df = df.loc[df['Country Name'].isin(countries)] 
    df = df.loc[df['Indicator Code'].eq(ind)]
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name','Indicator Code'], var_name='Years')
    del df2['Country Code']
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code'],'Country Name').reset_index() 
    return df, df2

countries= ['Germany', 'Canada', 'Japan', 'France']

#Line Plot


df,df2 = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv','ER.PTD.TOTL.ZS')    
    
plt.figure(dpi=144)  
df2['Years'] = pd.to_numeric(df2['Years'])
df2.plot("Years", countries, title ='Terrestrial and marine protected areas (% of total territorial area)',legend ='leftbottom')
plt.ylabel('Terrestrial and marine protected areas') 
plt.xlim(2016,2021)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # legend 
plt.show()





#line plot


df, df2 = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv','ER.LND.PTLD.ZS')

plt.figure(dpi = 144) # plotting figure
df2['Years'] = pd.to_numeric(df2['Years'])
df2.plot("Years", countries, title ='Terrestrial protected areas (% of total land area)',legend ='leftbottom')
plt.ylabel('Terrestrial protected areas') 
plt.xlim(2016, 2021)
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()



# Pie chart

 
df, df2 = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv','SI.POV.DDAY')
df2.dropna()  

ger = np.sum(df2['Germany'])
cda = np.sum(df2['Canada']) # taking percentage
jap = np.sum(df2['Japan'])
fra = np.sum(df2["France"])

total = ger + cda + jap + fra

germany = ger/ total*100
canada = cda/ total*100
japan = jap/ total*100
france = fra/ total*100

pov_head = np.array([germany, canada, japan, france])
explode =(0.0,0.1,0.0,0.0)

plt.figure(dpi=144)
plt.pie(pov_head, labels = countries, shadow = True, explode = explode, autopct = ('%1.1f%%'))
plt.title("Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)") 
plt.show()




# Pie chart

df, df2 = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv','EG.ELC.ACCS.ZS')
df2.dropna()

ger = np.sum(df2['Germany'])
cda = np.sum(df2['Canada']) # taking percentage
jap = np.sum(df2['Japan'])
fra= np.sum(df2["France"])

total = ger + cda + jap + fra

germany = ger/ total*100
canada = cda/ total*100
japan = jap/ total*100
france = fra/ total*100

electricity = np.array([germany, canada, japan, france])
explode =(0.1,0.0,0.0,0.0) 

plt.figure(dpi = 144)
plt.pie(electricity, labels = countries, shadow = True, explode = explode, autopct = ('%1.1f%%'))
plt.title("Access to electricity (% of population)") 
plt.show()





# Bar chart



df, df2 = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv','EN.ATM.NOXE.KT.CE')
df2 = df2.loc[df2['Years'].isin(['2010','2001','2002','2003','2004'])]
df2.dropna()


num = np.arange(5)
width = 0.2
years = df2['Years'].tolist() # taking data in list format

  
plt.figure()
plt.title('Nitrous oxide emissions (thousand metric tons of CO2 equivalent)')
plt.bar(num,df2['Canada'], width, label ='Canada')
plt.bar(num+0.2, df2['Japan'], width, label ='Japan')
plt.bar(num-0.2, df2['Germany'], width, label='Germany')

plt.xticks(num, years)
plt.xlabel('Years')
plt.ylabel('Nitrous oxide emissions')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()





# Bar chart 

df, df2 = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv','EN.ATM.METH.KT.CE')
df2 = df2.loc[df2['Years'].isin(['2010','2011','2012','2013','2014'])]
df2.dropna()


num= np.arange(5)
width = 0.2
years = df2['Years'].tolist() # taking data in list format

  
plt.figure()
plt.title('Methane emissions (kt of CO2 equivalent)')
plt.bar(num,df2['Canada'], width, label='Canada')
plt.bar(num+0.2, df2['Japan'], width, label='Japan')
plt.bar(num-0.2, df2['Germany'], width, label='Germany')
plt.xticks(num, years) 
plt.xlabel('Years')
plt.ylabel('Methane emissions')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()



# Statistical Analysis


df_a, df_b = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv','EN.ATM.METH.KT.CE')
df_b = df_b[(df_b['Years']>="1990") & (df_b['Years']<="2020")]
df_b.dropna()

countries_mean = np.mean(df_b[countries])
countries_skew = stats.skew(df_b[countries])

print("countries_mean: ",countries_mean)
print("countries_skew: ",countries_skew)


