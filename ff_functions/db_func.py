import datetime as dt
import os
import pwd
import sqlite3

def get_username():
    """Get the username regardless of Windows/Mac OS

    Returns:
        str: Username of computer to create paths
    """    
    return pwd.getpwuid(os.getuid())[0]


def db_connect(db_name, db_path):
    """[summary]

    Args:
        db_name ([type]): [description]
        db_path ([type]): [description]

    Returns:
        [type]: [description]
    """    
    user = get_username()
    return sqlite3.connect(f'/Users/{user}/{db_path}/{db_name}.sqlite3')


def backup_db(db_name, db_path):
    """[summary]

    Args:
        db_name (str): Name of database to be copied for backup
        root_path (str, optional): Path to current database folder without /Users/{username}/
        (will be auto added). Defaults to '/Documents/Github/Fantasy_Football/Data/Databases/'.
    """    
    # get the username for current OS
    username = get_username()

    # setup the old and new paths to save out database
    today_time = dt.datetime.now().strftime('_%Y_%m_%d_%M')
    old_path = f'/Users/{username}/{db_path}/{db_name}.sqlite3'
    new_path = f'/Users/{username}/{db_path}/DB_Versioning/{db_name}_{today_time}.sqlite3'

    # copy the current database over to new folder with timestamp appended    
    os.copyfile(old_path, new_path)

    print(f'Backup save to {new_path}')


def append_to_db(df, db_name, table_name, db_path, if_exist='append'):
    """[summary]

    Args:
        df ([type]): [description]
        db_name ([type]): [description]
        table_name ([type]): [description]
        if_exist (str, optional): [description]. Defaults to 'append'.
        db_path (str, optional): [description]. Defaults to '/Documents/Github/Fantasy_Football/Data/Databases/'.
    """    
    # create backup of current database
    backup_db(db_name, db_path)

    # connect to main database and create or update table
    conn = db_connect(db_name, db_path)
    df.to_sql(name=table_name, con=conn, if_exists=if_exist, index=False)