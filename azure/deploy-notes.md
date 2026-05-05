# Déploiement sur Azure App Service (Linux)

## Prérequis
- Un compte Azure.
- Azure CLI installé.
- Le projet poussé sur GitHub.

## Étapes de déploiement
1. **Création du plan App Service :**
   Créez un plan App Service sous Linux avec Python 3.11.

2. **Création de la Web App :**
   Créez une Web App rattachée à ce plan.

3. **Variables d'environnement :**
   Allez dans `Configuration` > `Paramètres d'application` et ajoutez votre `GOOGLE_API_KEY`.

4. **Commande de démarrage (Startup Command) :**
   Configurez la commande de démarrage de l'App Service pour qu'elle pointe sur notre script :
   ```
   bash azure/startup.sh
   ```

5. **Déploiement GitHub Actions ou Local Git :**
   Déployez le code source. Attention : assurez-vous que la base ChromaDB (dossier `chroma_db/`) est soit générée lors du déploiement, soit poussée manuellement si les notes sont statiques (bien que non recommandé dans Git, un stockage Azure Blob Storage est préférable en production).
