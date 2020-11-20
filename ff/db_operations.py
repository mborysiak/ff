#%%
import datetime as dt
import os
import shutil
from ff.general import get_timestamp, get_username
import sqlite3
import pandas as pd 

class DataManage:

    def __init__(self, data_path):
        """Create DataManage class by passing path to main database directory

        Args:
            data_path (str): Path where main databases exist and backups are stored
        """        

        self.data_path = data_path


    def db_connect(self, db_name):
        """Connect to specific database in data_path

        Args:
            db_name (str): name of database to connect to

        Returns:
            sqlite3.connection: Connection to sqlite3 database
        """    
        return sqlite3.connect(f'{self.data_path}/{db_name}.sqlite3')


    def backup_db(self, db_name, backup_folder='DB_Versioning'):
        """Save timestamped backup of database

        Args:
            db_name (str): Name of database to be copied for backup
            backup_folder (str, optional): [description]. Defaults to 'DB_Versioning'.
        """        
        # get timestamp string to append to name
        ts = get_timestamp()

        # setup the old and new paths to save out database
        old_path = f'{self.data_path}/{db_name}.sqlite3'
        new_path = f'{self.data_path}/{backup_folder}/{db_name}_{ts}.sqlite3'

        # copy the current database over to new folder with timestamp appended    
        shutil.copyfile(old_path, new_path)

        # write out confirmation
        print(f'Backup saved to {new_path}')

    def read(self, q, db_name):
        """Read

        Args:
            q ([type]): [description]
            db_name ([type]): [description]

        Returns:
            [type]: [description]
        """        
        conn = self.db_connect(db_name)
        return pd.read_sql_query(q, conn)


    def write_to_db(self, df, db_name, table_name, if_exists):
        """Write data to table by creating or appending

        Args:
            df (pandas.DataFrame): DataFrame to write out to databse
            db_name (str): Name of database where table of interest exists
            table_name (str): Name of table to create or modify
            if_exists (str): 'append' or 'replace' arguments to pass to df.to_sql
        """    
        # create backup of current database
        self.backup_db(db_name)

        # connect to database and create or update table
        conn = self.db_connect(db_name)
        df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=False)
        conn.close()
        
        # write out confirmation
        print(f'{if_exists.title()}(ed) data to {db_name}.{table_name}' )


    def delete_from_db(self, db_name, table_name, where_state):
        """Delete records from table according to WHERE statement

        Args:
            db_name (str): Name of database to connect to
            table_name (str): Table name to delete records from
            where_state (str): Condition to pass to WHERE statement
        """  
        # create backup of current database
        self.backup_db(db_name)

        # connect to database and create cursor object
        conn = self.db_connect(db_name)
        cur = conn.cursor()

        # execute delete statement
        cur.execute(f'''DELETE FROM {table_name} 
                        WHERE {where_state}''')
        conn.commit(); conn.close()

        # write out confirmation
        print(f'Deleted data from {db_name}.{table_name} where {where_state}' )