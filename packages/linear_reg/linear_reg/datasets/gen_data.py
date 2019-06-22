import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_reg.configs import config

np.random.seed(1)

def gen_data(n=1001):
    """
    Generate training data
    
    Arguments:
        n -- number of sample
    
    Returns:
        df -- return dataframe
    """
    x = np.linspace(-100,100,n)
    rand = np.random.uniform(-20,20,len(x))
    y = 3*x + rand
    
    df = pd.DataFrame()
    df['X'] = x
    df['Y'] = y
    n_rand = int(n/10)
    rand_nan = np.random.choice(len(y),n_rand)
    df.loc[rand_nan,'X'] = np.nan
    
    fig = plt.figure(figsize=(8,5))
    plt.scatter(df['X'],df['Y'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data distribution')
    
    return df

def save_data(df, file_path = config.DATASET_DIR+'/'+'data.csv'):
    """
    Save the data
    
    Arguments:
        df -- dataframe for saving
        file_path -- path for saving
    """
    df.to_csv(file_path, index=False)
    

if __name__ == 'main':
    df = gen_data()
    save_data(df)
    
    