sudo apt-get update -y
sudo apt-get install aptitude -y
sudo aptitude update -y

sudo -H apt-get install build-essential -y
sudo -H apt-get install cmake git unzip zip -y
sudo -H apt-get install python3-dev python3-pip -y
sudo -H apt-get install tmux htop imagemagick -y

sudo -H pip2 install --upgrade pip
sudo -H pip3 install --upgrade pip
sudo -H pip3 install setuptools six numpy wheel mock psutil
sudo -H pip3 install Pillow scipy matplotlib==2.1.2

uname -m && cat /etc/*release
sudo apt-get install linux-headers-$(uname -r) -y

sudo apt-get purge nvidia* -y
sudo rm -rf /usr/local/cuda*

sudo sh /newNAS/Share/GPU_Server/cuda_9.0.176_384.81_linux.run --silent --toolkit
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib' >> ~/.bashrc
tar xf /newNAS/Share/GPU_Server/cudnn-9.0-linux-x64-v7.tgz -C /tmp
sudo cp -r /tmp/cuda/* /usr/local/cuda/

source ~/.bashrc
sudo ldconfig
nvidia-smi

sudo -H pip2 uninstall tensorflow_gpu tensorflow -y
sudo -H pip3 uninstall tensorflow_gpu tensorflow -y

sudo -H pip3 install /newNAS/Share/GPU_Server/tensorflow_gpu-1.5.0rc1-cp35-cp35m-manylinux1_x86_64.whl

sudo aptitude upgrade -y 
sudo apt-get upgrade -y 
sudo apt-get autoremove -y
sudo apt-get autoclean