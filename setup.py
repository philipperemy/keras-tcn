from setuptools import setup

setup(
    name='keras-tcn',
    version='2.1.0',
    description='Keras TCN',
    author='Philippe Remy',
    license='MIT',
    packages=['tcn'],
    # manually install tensorflow or tensorflow-gpu
    install_requires=['numpy',
                      'keras']
)
