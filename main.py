import pandas as pd
from fmm import FMM

if __name__=="__main__":
    df= pd.read_csv('Opt_dataset_origin.csv',header=None)
    fmm = FMM(df,z=15,w=10)
    fmm.train()

