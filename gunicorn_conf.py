import os

workers = int(os.environ.get("GUNICORN_WORKERS", "4"))
threads = int(os.environ.get("GUNICORN_THREADS", "1"))
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:9875")
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "uvicorn.workers.UvicornWorker")
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
accesslog = os.environ.get("GUNICORN_ACCESSLOG", "-")
errorlog = os.environ.get("GUNICORN_ERRORLOG", "-")
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "120"))
forwarded_allow_ips = "*"
secure_scheme_headers = {"X-FORWARDED-PROTO": "https"}
reload = bool(os.environ.get("GUNICORN_RELOAD", "False"))
preload_app = True
