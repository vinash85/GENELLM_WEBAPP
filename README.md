# GENELLM_WEBAPP

## Docker Compose usage GUIDE
1. clone the git repo
2. go to the docker directory "cd GENELLM_WEBAPP/requirements/docker"
3. start the docker container on a server with command : "docker-compose -f app.yml up -d"
NOTE: the docker compose  orchestration will always try to restart the container in case of any failure. Use step 5 to debug. 
4. Go to the browser and type ip address followed by port number(5000 default for flask): http://w.x.y.z:5000
5. To check the logs of docker: "docker-compose -f app.yml logs"
6. To bring down the container: "docker-compose -f app.yml down"

