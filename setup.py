from setuptools import setup, find_packages

setup(
    name='datos_lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels'
    ],
    author='Bechi',
    description='Libreria para analisis de datos y regresion',
    url='https://github.com/BechiixD/datos_lib',
)