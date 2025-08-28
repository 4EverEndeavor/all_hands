from setuptools import setup, find_packages

setup(
    name='all-hands',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'ollama',
        'requests',
    ],
    entry_points={
        'console_scripts': [
            'all-hands = main_agent:main',
        ],
    },
)
