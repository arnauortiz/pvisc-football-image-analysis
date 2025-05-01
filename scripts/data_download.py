import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Cargar las variables del archivo .env
load_dotenv()

# Leer la API key desde la variable de entorno
api_key = os.getenv("ROBOFLOW_API_KEY")

# Conectar con Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace("roboflow-jvuqo").project("football-players-detection")
dataset = project.version(1).download("yolov8")
