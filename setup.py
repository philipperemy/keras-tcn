from setuptools import setup

setup(
    name='keras-tcn',
    version='1.8.0',
    description='Keras TCN',
    author='Philippe Remy',
    license='MIT',
    packages=['tcn'],
    install_requires=['tensorflow',
                      'numpy',
                      'keras']
)
