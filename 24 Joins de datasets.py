import pandas as pd 

filepath = "python-ml-course-master/datasets/athletes/"
data_main = pd.read_csv(filepath + 'Medals.csv', encoding="ISO-8859-1")
a = data_main['Athlete'].unique().tolist()
len(a)
data_main.shape
data_country = pd.read_csv(filepath + 'Athelete_Country_Map.csv', encoding= 'ISO-8859-1')

len(data_country)
data_country[data_country['Athlete']=='Aleksandar Ciric']

data_sport = pd.read_csv(filepath + 'Athelete_Sports_Map.csv', encoding='ISO-8859-1')
data_country_dp = data_country.drop_duplicates(subset='Athlete')

data_main_country = pd.merge(left= data_main, right = data_country_dp, left_on ="Athlete", right_on="Athlete")

data_sport_dp = data_sport.drop_duplicates(subset='Athlete')

data_final = pd.merge(left=data_main_country, right=data_sport_dp,left_on="Athlete",right_on="Athlete")
print(data_final.head(),data_final.shape)

#Merge can have a parameter 'how' to specify the type of the join (inner, left,rigth,outer)