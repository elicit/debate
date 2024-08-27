reorder-python-imports --py311-plus $(find . -name '*.py')
black .
flake8 . --ignore=E501,W503,E704,E203
pyright .