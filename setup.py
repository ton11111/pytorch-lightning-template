from setuptools import find_namespace_packages, setup


setup(
    install_requires=[
        "lightning==2.3.2",
        "torchmetrics==1.4.0",
        "torchvision==0.18.1",
    ],
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    test_suite="tests",
)
