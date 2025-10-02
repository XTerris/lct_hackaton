#!/usr/bin/env bash

gunicorn -c /app/gunicorn_conf.py main:app
