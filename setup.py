from setuptools import setup

setup(
    name='cc_model',
    version='0.1.0',
    packages=['cc_model'],
    author='SomeOne',
    author_email='some_mail@some_where.com',
    description='This package enables the colorful-configuration model for python',
    install_requires=[
              'pandas', 'scipy', 'numpy', 'matplotlib', 'numba', 'networkx'
          ],
    python_requires='>=3.8'
)
