#
from my_package.sub_package.config import DATABASE_URL
from my_package.main import APP_NAME
#
#from .config import DATABASE_URL
#from ..main import APP_NAME

def get_db_info():
    return f'{APP_NAME} connects to {DATABASE_URL}'


