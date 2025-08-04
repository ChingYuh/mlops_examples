
import pandas as pd

def create_sample_dataframe():
    """Creates a sample Pandas DataFrame."""
    data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
    df = pd.DataFrame(data)
    print("Sample DataFrame created.")
    return df

if __name__ == '__main__':
    create_sample_dataframe()
