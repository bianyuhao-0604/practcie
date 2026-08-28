import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_dir)

from my_package.sub_package.utils import get_db_info
print(get_db_info())