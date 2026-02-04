#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting installation of Docker, Docker Compose, Anaconda, and Python...${NC}"

# 1. Update and Install Dependencies
echo -e "${GREEN}Updating system and installing prerequisites...${NC}"
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg wget

# 2. Install Docker and Docker Compose
echo -e "${GREEN}Setting up Docker repository...${NC}"
sudo install -m 0755 -d /etc/apt/keyrings
# Download Docker's official GPG key
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo -e "${GREEN}Installing Docker Engine and Docker Compose...${NC}"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group
echo -e "${GREEN}Adding user '$USER' to the docker group...${NC}"
sudo usermod -aG docker "$USER"

# 3. Install Anaconda (includes Python)
ANACONDA_VERSION="2025.12-2"
INSTALLER_NAME="Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh"
DOWNLOAD_URL="https://repo.anaconda.com/archive/${INSTALLER_NAME}"
INSTALL_PATH="$HOME/anaconda3"

echo -e "${GREEN}Downloading Anaconda installer (${ANACONDA_VERSION})...${NC}"
wget "$DOWNLOAD_URL" -O anaconda_installer.sh

echo -e "${GREEN}Installing Anaconda to ${INSTALL_PATH}...${NC}"
# -b: run in batch mode (no manual intervention), -p: installation prefix
bash anaconda_installer.sh -b -p "$INSTALL_PATH"

echo -e "${GREEN}Initializing Conda for bash...${NC}"
"$INSTALL_PATH/bin/conda" init bash

# Cleanup
echo -e "${GREEN}Cleaning up...${NC}"
rm anaconda_installer.sh

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}Installation Complete!${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "1. Docker and Docker Compose are installed."
echo -e "2. Anaconda (Python) is installed at $INSTALL_PATH."
echo -e ""
echo -e "${GREEN}IMPORTANT:${NC} Please run the following command (or restart your installation) to apply the group changes and path updates:"
echo -e "   ${BLUE}source ~/.bashrc && newgrp docker${NC}"
