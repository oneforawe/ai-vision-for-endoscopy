from setuptools import setup

# References:
# https://python-packaging.readthedocs.io/en/latest/minimal.html
# https://uoftcoders.github.io/studyGroup/lessons/python/packages/lesson/

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ai_vision_for_endoscopy',
    version='0.1',
    description='Package for developing neural networks for endoscopy.',
    url='https://github.com/oneforawe/ai-vision-for-endoscopy',
    author='Andrew W. Forrester',
    author_email='Andrew@Andrew-Forrester.com',
    license='MIT',
    packages=['ai_vision_for_endoscopy'],
    install_requires=requirements,
    # long_description=open('README.md').read(),
    #zip_safe=False
)

