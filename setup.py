from setuptools import setup, find_packages
import re

NAME = "torch_nl"

# read the version number from the library
pattern = r"[0-9]\.[0-9]"
VERSION = None
with open("./torch_nl/__init__.py", "r") as fp:
    for line in fp.readlines():
        if "__version__" in line:
            VERSION = re.findall(pattern, line)[0]
if VERSION is None:
    raise ValueError("Version number not found.")


with open("requirements.txt") as f:
    install_requires = list(
        filter(lambda x: "#" not in x, (line.strip() for line in f))
    )

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    zip_safe=True,
    python_requires=">=3.8",
    license="MIT",
    author="Fe" + "\u0301" + "lix Musil",
    install_requires=install_requires,
)
