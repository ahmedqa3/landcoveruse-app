# Utilisez une image de base Python
FROM python:3.8-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements et installer les dépendances
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copier le reste du code de l'application
COPY . .

# Exposez le port 5000 (ou celui que vous utilisez)
EXPOSE 5000

# Commande pour lancer l'application
CMD ["python", "appl.py"]
