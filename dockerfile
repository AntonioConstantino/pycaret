# Usa la imagen base con Python 3.10
FROM python:3.10-slim

# Configuración básica
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install  -r requirements.txt

# Instalar nbconvert para convertir notebooks
RUN pip install nbconvert

# Copiar los archivos del proyecto
COPY . .

# Convertir automáticamente el notebook a script
RUN jupyter nbconvert --to script Scripts/ml.ipynb

# Establecer el archivo convertido como entrada
CMD ["python", "Scripts/ml.py"]
