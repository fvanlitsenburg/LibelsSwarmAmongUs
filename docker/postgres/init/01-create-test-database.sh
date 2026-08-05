#!/usr/bin/env bash

set -euo pipefail

psql \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  --set ON_ERROR_STOP=1 <<SQL
CREATE DATABASE lsau_test OWNER "$POSTGRES_USER";
SQL