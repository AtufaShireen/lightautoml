from setuptools import setup, find_packages

setup(name='light-automl',
    version='0.2',
    packages=find_packages(where="automl"),
    author='Atufa Shireen',
    package_dir={"": "automl"},
    url = "https://github.com/AtufaShireen/lightautoml",
    
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
  ],
  install_requires=[
'joblib==1.1.0',
'kneed==0.7.0',
'matplotlib==3.5.0',
'numpy==1.21.4',
'pandas==1.3.4',  
'scikit-learn==0.24.2',
'scikit-optimize==0.9.0',
'scikit-plot==0.3.7',
'scipy==1.7.3',
'xgboost==1.5.0',

  ])
