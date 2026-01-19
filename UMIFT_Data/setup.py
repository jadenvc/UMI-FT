from setuptools import setup, find_packages

INSTALL_REQUIRES = []

setup(
    name="umift",
    author="Chuer Pan",
    version="0.0.0",
    description="UMI Force Torque",
    long_description=open('README.md').read(),
    keywords=[],
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=[],
    zip_safe=False
)
