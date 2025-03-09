from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def  get_requirements(file_path : str)->List[str]:
    """
    this funcation will return the list of requirements

    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()

        # when we use readlines ,it will read the requriments.txt line by line .
        # will also include "\n" to represent newline.
        # replace "\n"-->""
        requirements=[req.replace('\n','') for req in requirements]

        # remove "-e ." from requirements.txt:
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="ML PROJECT",
    version="0.0.1",
    author="Owais Sofi",
    author_email="owaissofi3351@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)