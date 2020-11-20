#%%

import pandas as pd

def name_clean(player_name):
    """Cleans up player names from special characters to ensure 
    matching between data sources

    Args:
        player_name (str): player name to be cleaned

    Returns:
        [str]: player name stripped of special characters,
               white space, and title cases

    Test:
        >>> name_clean('JuJu Smith-Schuster II. Jr**%')
        'Juju Smith Schuster'
    """    

    # replace common characters to remove from names with no space
    characters = ['*', '+', '%', ',' ,'III', 'II', '.', 'Jr']
    for c in characters:
        player_name = player_name.replace(c, '')
        
    # replace dashes with space and title case / strip whitespace
    player_name = player_name.replace('-', ' ')
    player_name = player_name.title().rstrip().lstrip()
    
    return player_name

def convert_to_float(df):
    """Convert all columns in dataframe to float (if possible)

    Args:
        df (pandas.DataFrame): pandas dataframe to be converted

    Returns:
        pandas.Dataframe: dataframe with all possible columns converted to float
    
    Test:
        >>> pd.DataFrame({'A': ['a', 'b'], 
        ...               'B': ['3', '2']})
            A  B
            0  a  3
            1  b  2
    """    
    for c in df.columns:
        try: df[c] = df[c].astype('float')
        except: pass
    return df


def remove_dup_cols(df):
    """Remove duplicated columns in dataframe

    Args:
        df (pandas.DataFrame): DataFrame with duplicated columns to condense

    Returns:
        pandas.DataFrame: DataFrame with duplicated columns removed
    """    
    return df.loc[:,~df.columns.duplicated()]
