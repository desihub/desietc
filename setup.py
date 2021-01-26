from setuptools import setup, find_packages

setup(
    name='desietc',
    version='0.1.dev0',
    description='Online exposure-time calculator for DESI',
    url='http://github.com/dkirkby/desietc',
    author='David Kirkby',
    author_email='dkirkby@uci.edu',
    license='MIT',
    packages=find_packages(exclude=["tests",]),
    install_requires=['numpy', 'scipy'],
    include_package_data=False,
    zip_safe=False,
    entry_points = {
        'console_scripts': [
            'etcoffline=desietc.scripts.offline:main',
            'etcdepth=desietc.scripts.depth:main',
        ],
    }
)
