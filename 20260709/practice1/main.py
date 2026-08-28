import sys
import practice1.my_module as my_module

print(f"调用1: {my_module.increment()}")  # 1
print(f"调用2: {my_module.increment()}")  # 2

import practice1.my_module as my_module
print(f"调用3: {my_module.increment()}")

print(f"是同一对象: {id(sys.modules['my_module']) == id(my_module)}")

import importlib
importlib.reload(my_module)
print(f"重载后: {my_module.increment()}")