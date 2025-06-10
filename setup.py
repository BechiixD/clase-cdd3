from setuptools import setup, find_packages

setup(
    name='datos_lib',
    version='0.1.0',  # Usando SemVer
    packages=find_packages(),  # Detecta datos_lib y datos_lib.regresion
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'scipy>=1.5.0',
        'matplotlib>=3.3.0',
        'statsmodels>=0.12.0',
    ],  # Incluye versiones mínimas para estabilidad
    author='Bechi',
    author_email='tu_email@example.com',  # Opcional
    description='Biblioteca de Python para análisis descriptivo, generación de datos y regresión lineal/logística',
    long_description=open('README.md').read(),  # Incluye el README como descripción larga
    long_description_content_type='text/markdown',  # Especifica formato Markdown
    url='https://github.com/BechiixD/datos_lib',
    license='MIT',  # Cambia según tu elección
    python_requires='>=3.8',  # Coincide con el README
    classifiers=[  # Ayuda a categorizar el paquete en PyPI
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)