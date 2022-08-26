from setuptools import setup

setup(
    name='deepfa',
    version='0.1.0',    
    description='DeepFA',
    url='https://github.com/barbarabenato/DeepFA.git',
    author='Bárbara C. Benato',
    author_email='barbara.benato@ic.unicamp.br',
    license='BSD 2-clause',
    packages=['deepfa'],
    install_requires=[
                      'numpy',
                      'scipy',
                      'matplotlib',
                      'scikit-learn', 
                      'tensorflow'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)