Please use a Linux enviroment for executing code. Windows tents to be way buggier than linux.
Save the software preferably into your home directory


When on linux, python should be already installed, so check version.
Else...Install python 3.6 <= Version <= 3.9
#on different OS
https://realpython.com/installing-python/
#on Debian/Ubuntu
https://linuxize.com/post/how-to-install-python-3-9-on-ubuntu-20-04/


Check if python is the right version
python --version
#sometimes its called python3 or python3.7 python3.9


Install python package manager
Open terminal & execute command:
python -m ensurepip --upgrade
#If you installed python3.9 the commands are python3.9 and for pip= pip3.9
#Also Check if pip (python package manager is the right python version

pip --version
pip3 --version
pip3.9 --version


Install Jupyter-lab to be able to open the jupyter-notebook
#!!! ATTENTION!!! its really important, that the correct pip with corresponding python version is chosen, because jupyter lab will use the python version specified by the pip version

pip install jupyterlab

Now open Jupyter-lab by changing directory into the working directory of the Software
cd Alike-Fue
jupyter-lab

Now Jupyter-Lab should open and you can click on "Fractal GAN.ipynb"
You can execute cell by cell to install all necessary packages to execute the code.

Now you can read the work and execute some code, but beware... jupyter notebook is just for demonstration purposes.
If the output due to prints gets too long, jupyter notebook will crash and the file will be softbricked.
If you want to train/test/validate neural networks yourself, then execute "python ALIKE.py", 
which is the script based version of the code.



