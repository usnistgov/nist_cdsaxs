from setuptools import setup

# TODO: implement namespace packages for future separation of legacy gui

setup(name='cdsaxs_gui_legacy',
      version='0.4',
      description='A GUI for CDSAXS data processing',
      url='https://github.com/usnistgov/CDSAXSMCMCProject',
      author='Christopher Liman',
      author_email='christopher.liman@nist.gov',
      license='Public Domain',
      packages=['cdsaxs_gui_legacy'],
      package_dir={
          '': 'src',
      },
      classifiers=['Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.4'],
      install_requires=['future', 'futures', 'namedlist', 'pyfits', 'tifffile']
      )
