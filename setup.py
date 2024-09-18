from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="detector_benchmark",
    version="0.1.0",
    packages=find_packages(include=["detector_benchmark", "detector_benchmark.*"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "create_dataset=detector_benchmark.create_dataset:main",
            "test_detector=detector_benchmark.test_detector:main",
        ],
    },
    author="Henrique Da Silva Gameiro",
    author_email="henrique.dasilvagameiro@gmail.com",
    description="A package for LLM detector benchmarking",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/marluxiaboss/benchmark_ai_news_detection",
)
