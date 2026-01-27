#!/usr/bin/env bash
set -e  # exit immediately if a command fails

sudo apt update && sudo apt upgrade -y && sudo apt install nginx -y
sudo apt install certbot python3-certbot-nginx -y
rm -fr q_easy
git clone https://github.com/Joeonome/q_easy.git
cd q_easy
sudo apt install python3.12-venv -y

python3.12 -m venv myenv
source myenv/bin/activate
pip install --no-cache-dir -r requirements.txt
sudo apt install -y libgomp1

sudo npm install -g pm2
pm2 start "streamlit run app.py" --name q-easy
pm2 save
pm2 startup
