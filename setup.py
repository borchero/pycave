import os
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='pycave',
    version=os.getenv('CIRCLE_TAG'),

    author='Oliver Borchert',
    author_email='borchero@icloud.com',

    description='Well-Known Machine Learning Models in PyTorch with Strong GPU Acceleration.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',

    url='https://github.com/borchero/pycave',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,

    license='License :: OSI Approved :: MIT License',
    zip_safe=False
)
