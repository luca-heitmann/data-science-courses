#!/usr/bin/env bash

docker rm -f melanom_db

docker run -d \
  --name melanom_db \
  -e MARIADB_DATABASE=melanom_db \
  -e MARIADB_ROOT_PASSWORD=melanom_db \
  -v $(pwd):/app \
  -p 3306:3306 \
  mariadb

sleep 5

docker exec -it melanom_db bash -c "mariadb -u root --password='melanom_db' melanom_db < /app/create-dbs.sql"
docker exec -it melanom_db bash -c "mariadb -u root --password='melanom_db' melanom_db < /app/data-import.sql"
