#Carga de datos a través de la función open de python para datasets excesivamente grandes
#Crea un for que lee linea por linea del fichero y elimina aquellas que termina de processar para no quedarse sin memoria
import pandas as pd

mainpath = "C:/Developer/Data science y machine learning course/python-ml-course-master/datasets"
data = open(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", "r")
cols = data.readline().strip().split(",")
n_cols = len(cols)
counter = 0
main_dict = {}
for col in cols: 
    main_dict[col] =[]
for line in data:
    values = line.strip().split(",")
    for i in range (len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1
#print("El data set tiene %d filas y %d columnas"%(counter, n_cols))
#print(main_dict)
df = pd.DataFrame(main_dict)
inputfile = mainpath + "/" + "customer-churn-model/Customer Churn Model.txt"
outputfile = mainpath + "/" + "customer-churn-model/Tab Customer Churn Model.txt"
with open(inputfile, "r") as inputfile1:
    with open(outputfile,"w") as outputfile1:
        for line in inputfile1:
            fields = line.strip().split(",")
            outputfile1.write("\t".join(fields))
            outputfile1.write("\n")
df= pd.read_csv(outputfile, sep="\t")
print(df.head())
#print(df)