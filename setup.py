from setuptools import setup

setup(
    name='keras-tcn',
    version='2.9.2',
    description='Keras TCN',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['tcn'],
    # manually install tensorflow or tensorflow-gpu
    install_requires=[
        'numpy==1.16.2',
        'keras',
        'gast==0.2.2'
    ]
)
