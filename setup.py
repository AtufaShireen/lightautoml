from setuptools import setup, find_packages

setup(name='automl',
    version='1.0',
    packages=find_packages(where="automl"),
    author='Atufa Shireen',
    package_dir={"": "automl"},
    
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
  ],)
