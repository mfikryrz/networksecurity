### Network Security Projects For Phising Data (masukan semua parameter ini ke dalam Secrets and variables: Actions)

Setup github secrets:
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = 644669979862.dkr.ecr.us-east-1.amazonaws.com/networksecurit
ECR_REPOSITORY_NAME = networksecurit


Docker Setup In EC2 commands to be Executed, please enter this comman in EC2 Instance bash command
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker