import getpass
import datetime as dt

def get_username():
    """Get the username regardless of Windows/Mac OS

    Returns:
        str: Username of computer to create paths
    """    
    return getpass.getuser()


def get_timestamp():
    return dt.datetime.now().strftime('_%Y_%m_%d_%M')


def get_main_path(folder):
    user = get_username()
    return f'/Users/{user}/Documents/Github/{folder}/'
