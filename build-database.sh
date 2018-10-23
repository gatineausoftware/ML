#!/usr/bin/env bash

kubectl apply -f mysql-volumes.yaml
kubectl apply -f mysql-service-local.yaml



#   kubectl apply -f ../dev/${1}-local.yaml
#   Wait for all services to be deployed. Then run scripts to load
while [ $(kubectl get pods|grep ContainerCreating -c) != "0" ]; do   echo "Waiting for all deployments..." && sleep 10; done

#   Grab MySQL pod identifier
MYSQL_POD=$(kubectl get pods | grep -o "mysql-[a-zA-Z0-9]*-[a-zA-Z0-9]*")


kubectl cp data/heart.csv ${MYSQL_POD}:/var/lib/mysql-files/.
kubectl cp schema.sql ${MYSQL_POD}:/var/lib/mysql-files/.


#   Execute schema creation and data load on MySQL
#kubectl exec -it ${MYSQL_POD} -- bash -c "mysql -pfincrime --execute='CREATE USER \"tdai\" IDENTIFIED BY \"fincrime\"; GRANT ALL ON *.* TO \"tdai\";'"
#echo ${CYAN}"FILES LOADED! Creating users.."${RESTORE}
#DELETE FROM mysql.user WHERE User = "tdai";
#kubectl exec -it ${MYSQL_POD} -- bash -c "cd /var/lib/mysql-files/ && mysql -pfincrime --execute='source mysql_schemas.sql; source mysql-data-load.sql;'"
