from setuptools import find_packages, setup


HYPHEN_DOT_E = "-e ."
def get_requirements_details(filename):

    with open(filename,"r") as file:
        requirements = file.read().splitlines()

    if HYPHEN_DOT_E in requirements:

        requirements.remove(HYPHEN_DOT_E)
   
    return requirements


setup(
    name= "Credit Risk Assessment App",
    version="0.0.0.1",
    description="This app will help the company to predict whether the given user will default on loan or not",
    long_description='''This app is designed to predict the loan defulter. 
                        In this app, we will take input from the users on number of parameters and then predict 
                        whether he/she will default on the loan or not. This app uses the machine learning model and 
                        neural network to predict the result''',
    author="Deepak Pawar",
    author_email="deepakpw234@gmail.com",
    packages=find_packages(),
    install_req = get_requirements_details("requirements.txt")
)



