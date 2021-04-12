from setuptools import setup, find_packages

setup(name='PyMLE',
      version='0.0.1',
      description='Maximum Likelihood Estimation (MLE) and simulation for SDE',
      long_description='Maximum Likelihood Estimation (MLE) and simulation for '
                       'Stochastic Differential Equations (SDE)',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering ',
      ],
      keywords='sde mle maximum likelihood difussion estimation simulation',
      url='http://github.com/...',
      author='Justin Lars Kirkby',
      author_email='jkirkby33@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
      ],
      include_package_data=True,
      zip_safe=False)
