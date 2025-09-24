from setuptools import setup, find_packages

with open("requirements.txt") as rf:
    requirements = rf.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flipkart_product_recommendation",
    version="0.1.0",
    description="End-to-end Flipkart Product Recommendation System using LangChain, HuggingFace, Groq, AstraDB, and containerized deployment.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Bhopindra Parmar",
    author_email="bhupenparmar.ds@gmail.com",
    url="https://github.com/bhupencoD3/flipkart-product-recommendation",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
