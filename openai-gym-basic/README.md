#kreiranje enviromenta

conda create -n NAZIV_ENV python=3.7 anaconda

#ulazak u enviroment

conda activate NAZIV_ENV

#instalacija potrebnih biblioteka

conda install -c conda-forge gym

pip install tensorflow==1.15.0

pip install pycodestyle==2.7.0
pip install pyflakes==2.3.0
pip install flake8==3.9.0

conda install -c conda-forge gcc_linux-64
conda install -c conda-forge gcc_impl_linux-64

pip install stable-baselines[mpi] #ako bude problem


#registracija/aktiviranje gym-a...
#cd u foldergde je setup.py(mora se tako zvati fajl)

pip install -e .


#pokretanje test fajla

python gym_basic_env_test.py
