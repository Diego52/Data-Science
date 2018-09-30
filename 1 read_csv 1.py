#Carga de datos a través de la función read_csv

import pandas as pd 
import numpy as np

data =pd.read_csv("python-ml-course-master/datasets/titanic/titanic3.csv", sep=",", dtype={"a":np.float64},header=0, names={"ingresos","edad"}, skiprows=12, index_col=None, skip_blank_lines=False,na_filter=False)
