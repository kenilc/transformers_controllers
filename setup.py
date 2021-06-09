from setuptools import setup, find_packages

setup(
    name='transfomers_controllers',
    version='0.0.4',
    author='Kenil Cheng',
    author_email='<kenilc@gmail.com>',
    description='Helpers to control the text generation output by Hugging Face Transformers',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/kenilc/transformers_controllers',
    packages=find_packages(),
    install_requires=[
        'pygtrie>=2.4.2',
        'torch>=1.8.1',
        'transformers>=4.5.1'
    ],
    python_requires='>=3.6.0',
    keywords=[
        'python',
        'NLP',
        'deep learning',
        'transformer',
        'pytorch',
        'GPT',
        'GPT-2'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
