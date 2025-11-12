# iot_demo

# Update system
sudo apt update && sudo apt upgrade -y

# Install build dependencies
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev tk-dev uuid-dev wget

# Go to /usr/src to keep things tidy
cd /usr/src

# Download Python 3.11.6 (latest 3.11.x as of now)
sudo wget https://www.python.org/ftp/python/3.11.6/Python-3.11.6.tgz

# Extract the archive
sudo tar -xzf Python-3.11.6.tgz
cd Python-3.11.6

# Configure build with optimizations
sudo ./configure --enable-optimizations

# Compile using all CPU cores (takes 15–30 min on a Pi 4)
sudo make -j$(nproc)

# Install safely (altinstall prevents overwriting system python3)
sudo make altinstall

# Check installation
python3.11 --version

# Install pip and venv (should already be built-in, but just in case)
python3.11 -m ensurepip
python3.11 -m pip install --upgrade pip setuptools wheel
