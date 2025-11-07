#!/usr/bin/env bash

docker run -d --name melanom_db --env MARIADB_DATABASE=melanom_db --env MARIADB_ROOT_PASSWORD=melanom_db -p 3306:3306 mariadb
