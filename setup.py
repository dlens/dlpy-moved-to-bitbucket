'''
@author: Bill Adams
'''

from setuptools import setup

setup(name='dlpy',
      version='0.1',
      description='Decision Lens Common Tools Library',
      url='https://github.com/dlens/dlpy',
      author='Bill Adams',
      author_email='wjadams@decisionlens.com',
      license='MIT',
      package_dir={'dlpy':'dlpy'},
      python_requires=">=3.6",
      install_requires=['numpy', 'pandas'],
      packages=['dlpy'],
      #py_modules=['fxmodel'],
      #package_data={'bamath': ['data/*.csv', 'data/*.pickle', 'data/*.json', '*.css']},
      zip_safe=False)