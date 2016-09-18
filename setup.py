from setuptools import setup, find_packages

# Get version from tncontract/version.py
exec(open("tncontract/version.py").read())

setup(
    name = "tncontract",
    version = __version__,
    packages = find_packages(),
    author = "Andrew Darmawan",
    license = "MIT",
    install_requires = ["numpy", "scipy"],
)
