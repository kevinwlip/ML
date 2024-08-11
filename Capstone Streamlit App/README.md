
README
======

This is a Financial Sentimental Prediction Application created through 
Streamlit and hosted on Amazon EC2

Machine Learning Models created by myself and are hosted on
HuggingFace - https://huggingface.co/kevinwlip

GitHub - https://github.com/kevinwlip


Deployment - Streamlit with AWS EC2
-----------------------------------

Sources:
1. Deploying an OpenAI Streamlit Python App on AWS EC2 - https://www.youtube.com/watch?v=oynd7Xv2i9Y
2. AWS Tutorials: Deploy Python Application on AWS EC2 - https://www.youtube.com/watch?v=3sQhVKO5xAA&t=141s


Have app contents on GitHub.

Connect to AWS EC2 Instance.

In EC2 command line:

$ sudo su
$ yum update
$ yum install git (first time)
$ yum install python3-pip

Get GitHub link to repo.

$ git clone [GitHub repo link]
$ cd [repo]
$ python3 -m pip install -r requirements.txt

# If issue with requests library try the following
$ python3 -m pip install --ignore-installed streamlit

$ python3 -m streamlit run [app.py]

# Keep app running, even if you close the EC2 instance window/terminal
$ nohup python3 -m streamlit run [app.py]

# Look for process ID and kill the app, to prevent AWS charges
$ sudo su
$ ps -ef
$ kill [PID]
