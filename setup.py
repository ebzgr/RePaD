from setuptools import setup, find_packages

# Load all dependencies from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='repad',
    version='0.1.0',
    description='Recursive Partitioning Algorithm for Dynamic Discrete Choice Models',
    author='Ebrahim Barzegary',
    author_email='ebzgry@gmail.com',
    url='https://github.com/ebzgr/RePaD',
    packages=find_packages(),
    install_requires=required,  # full list of packages
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.7',
)
