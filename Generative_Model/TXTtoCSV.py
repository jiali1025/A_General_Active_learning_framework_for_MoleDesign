import numpy as np 
import pandas as pd 
   
txt = np.loadtxt('simple_10_23_test.txt', dtype=str, comments=None) 
txtDF = pd.DataFrame(txt) 
txtDF.to_csv('simple_10_23_test.csv',index=False)

