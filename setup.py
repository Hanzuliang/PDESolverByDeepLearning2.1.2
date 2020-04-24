# -*- coding:utf-8 -*-
try:
    from setuptools import setup, find_packages
except:
    from distutils.core import setup
from codecs import open
from os import path

#版本号
VERSION = '2.1.2'

#发布作者
AUTHOR = "Zuliang Han"

#邮箱
AUTHOR_EMAIL = "1461790569@qq.com"

#项目网址
URL = "https://github.com/Hanzuliang/PDESolverByDeepLearning"

#项目名称
NAME = "PDESolverByDeepLearning"

#项目简介
DESCRIPTION = "This package is suitable for solving the problem of one-dimensional n-order differential equation with Dirichlet boundary conditions."

#LONG_DESCRIPTION为项目详细介绍，这里取README.md作为介绍
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.txt'), encoding='ISO-8859-1') as f:
    LONG_DESCRIPTION = f.read()

#搜索关键词
KEYWORDS = ["Deep Learning", "Machine Learning", "Neural Networks", "Scientific computing", "Differential equations", "PDE solver"]

#发布LICENSE
LICENSE = "MIT"

#包
PACKAGES = ["PDESolverByDeepLearning"]

#具体的设置
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',

    ],
    #指定控制台命令
    entry_points={
        'console_scripts': [
            'PDESolver = PDESolverByDeepLearning:PDESolver',
        ],
    },
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    install_requires=['matplotlib', 'numpy', 'tensorflow'],                        #依赖的第三方包
    include_package_data=True,
    zip_safe=True,
)