#
# Created by maks5507 (me@maksimeremeev.com)
#

from setuptools import setup, find_packages
import setuptools.command.build_py as build_py


setup_kwargs = dict(
    name='deepbooster',
    version='0.0.1',
    packages=['deepbooster'],
    install_requires=[
        'torch',
        'typing',
        'rmq_interface'
    ],
    setup_requires=[
    ],

    cmdclass={'build_py': build_py.build_py},
)

setup(**setup_kwargs)
