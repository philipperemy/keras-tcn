from setuptools import setup

setup(
    name='keras-tcn',
    version='2.9.0',
    description='Keras TCN',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['tcn'],
    # manually install tensorflow or tensorflow-gpu
    install_requires=[
        'numpy==1.17.3',
        'keras',
        'gast==0.2.2'
    ]
)
