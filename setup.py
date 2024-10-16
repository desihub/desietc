from setuptools import setup, find_packages


setup(
    name="desietc",
    version="0.1.19",  # MAKE SURE THIS MATCHES desietc/_version.py! There are also ways to automate this.
    description="Online exposure-time calculator for DESI",
    url="http://github.com/desihub/desietc",
    author="David Kirkby",
    author_email="dkirkby@uci.edu",
    license="MIT",
    packages=find_packages(
        exclude=[
            "tests",
        ]
    ),
    install_requires=["numpy", "scipy", "fitsio"],
    include_package_data=False,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "etcreplay=desietc.scripts.replay:main",
            "etcdepth=desietc.scripts.depth:main",
        ],
    },
)
