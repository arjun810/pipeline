from setuptools import setup

setup(
    name="ziang",
    packages=['ziang'],
    version = "0.1.0a",
    description = "Task pipeline system",
    install_requires=["networkx", "cloud"],
    dependency_links=["git+git://github.com/arjun810/lucidity.git"],
)
