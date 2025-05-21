import pandas as pd
import glob
files = glob.glob('./clean_data/CO2/*.xlsx')
aep_files = glob.glob('./clean_data/AEP/*.xlsx')

def melt_data(df, name):
    return df.melt(id_vars=['Country Name', 'Country Code'],
            var_name='Year',
            value_name=name)
def merge_data(file,df_final):
    name = file.split('\\')[-1].split('.')[0].lower()
    df = pd.read_excel(file)
    df_melt = melt_data(df,name)
    df_final = df_final.merge(df_melt, on=['Country Name','Country Code','Year'], how='left')
    return df_final

init = pd.read_excel(files[0])
init = init.melt(id_vars=['Country Name','Country Code'],
                     var_name='Year')
df_init = init.iloc[:,:3].copy()
df_pol = df_init
df_aep = df_init

for file in files:
    df_pol = merge_data(file,df_pol)
for file in aep_files:
    df_aep = merge_data(file,df_aep)

df_pol = df_pol.melt(id_vars=['Country Name','Country Code','Year'],
                     var_name='Pollutant')
df_aep = df_aep.melt(id_vars=['Country Name','Country Code','Year'],
            var_name='AEP')

df_final = df_pol.merge(df_aep, on=['Country Name','Country Code','Year'],)

df_final['Year'], year_mapping = df_final['Year'].factorize()
df_final['Year']+=1
df_final['Pollutant'],pollutant_mapping = df_final['Pollutant'].factorize()
df_final['Pollutant']+=1
df_final['AEP'], aep_mapping = df_final['AEP'].factorize()
df_final['AEP']+=1

def build_table(df,map,colname):
    dict = {}
    for obj in df:
        dict[int(obj)] = map[obj-1]
    table = pd.DataFrame.from_dict(dict, orient='index')
    table = table.reset_index().rename(columns={'index':'id',0:colname})
    return table

df_year = df_final['Year'].unique()
year_table = build_table(df_year,year_mapping,'Year')

df_pollutant = df_final['Pollutant'].unique()
pollutant_table = build_table(df_pollutant,pollutant_mapping,'Pollutant')

df_aep = df_final['AEP'].unique()
aep_table = build_table(df_aep,aep_mapping,'AEP')

df_final.rename(inplace=True,columns={'Country Name':'country_name','Country Code':'country_code','Year':'year','Pollutant':'pollutant_id','value_x':'pollutant_value','AEP':'economy_id','value_y':'economy_value'})
year_table.rename(inplace=True,columns={'Year':'year'})
pollutant_table.rename(inplace=True,columns={'id':'pollutant_id','Pollutant':'source'})
aep_table.rename(inplace=True,columns={'id':'economy_id','AEP':'source'})

pollutant_table['pollutant'] = pollutant_table['source'].apply(lambda x: x.split('_')[0].upper())

list = pollutant_table['source'].apply(lambda x: max(x.split('_'), key=len))
dict_categorize = {}
for i in list:
    building = ['buildings','building']
    industrial_aera = ['industrial','viomixania']
    transport = ['transport','metakinhsewn']
    if i in building:
        dict_categorize[i]='Buildings (Mt)'
    elif i in industrial_aera:
        dict_categorize[i]='Industrial areas (Mt)'
    elif i in transport:
        dict_categorize[i]='Transport (Mt)'
    elif i == 'material':
        dict_categorize[i]='Waste material (Mt)'
    elif i == 'industry':
        dict_categorize[i]='Power industry (Mt)'
    elif i == 'capita':
        dict_categorize[i]='Capita (Tons)'
    elif i == 'pollution':
        dict_categorize[i]='Smoke from natural sources,cars,industry (Mt)'
def categorize(text):
    if text in dict_categorize.keys():
        return dict_categorize[text]
pollutant_table['source'] = pollutant_table['source'].apply(lambda x: max(x.split('_'), key=len)).apply(categorize)

from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://your_connection:password@your_host/database")

aep_table.to_sql('aep',engine,if_exists='replace',index=False)
year_table.to_sql('time',con=engine,if_exists='append',index=False)
pollutant_table.to_sql('pollutant',con=engine,if_exists='append',index=False)
df_pol.to_sql('fact',con=engine,if_exists='append',index=False)