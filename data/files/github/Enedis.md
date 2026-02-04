# M2 SISE - Enedis
> [!NOTE]
> Ce travail est un projet scolaire réalisé dans le cadre de notre 2<sup>ème</sup> année de Master en SISE.

<br>

## Présentation
### 🔗 Live : [France Energie](https://france-energie.koyeb.app/)
Visualisez la consommation des Français sur une carte et à l'aide de graphiques interactifs, en utilisant les données de l'API de **l'ADEME** et d'**Enedis**. Prédiction de la consommation et de l'étiquette DPE d'un logement grâce à un modèle de régression et de classification.
<br><br>
![Capture d'écran du site](screenshot.png)

<br>

## Technologies utilisées
- Backend : **Python** + **Flask**
- Frontend : **HTML** + **CSS** + **JS**
- Carte : **Leaflet** + plugins
- Graphiques : **Plotly**
- Gestion des données : **Pandas**
- Création des modèles : **scikit-learn**
- Déploiement : **Docker** + **Koyeb**

<br>

## Sources de données
Les données utilisées par l'application proviennent des APIs de l'[ADEME](https://data.ademe.fr/datasets/dpe03existant/api-doc) et d'[Enedis](https://data.enedis.fr/explore/dataset/consommation-annuelle-residentielle-par-adresse/api/). La base de données data.gouv [communes de France 2025](https://www.data.gouv.fr/datasets/communes-et-villes-de-france-en-csv-excel-json-parquet-et-feather/) est également exploitée.

<br>

## Exécuter l'app en local

1. Aller dans le dossier `web`
```bash
cd web
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Créer un dossier pour stocker le volume et renseigner son emplacement dans la variable d'environnement `MOUNT_PATH`
> macOS / linux
```bash
mkdir volume
export MOUNT_PATH="volume"
```
> Windows
```bash
mkdir volume
set MOUNT_PATH="volume"
```

4. Lancer l'app
```bash
python app.py
```

5. L'application est hébergée sur le port `8000`, ouvrir un navigateur et se rendre à l'adresse http://0.0.0.0:8000/

<br>

## Cahier des charges

![charge_1](https://github.com/user-attachments/assets/aa0d8716-70fc-40de-98e3-9a2aa5f333e5)
![charge 2](https://github.com/user-attachments/assets/177a3d8d-04e9-4ee7-b29c-0d9a7fc16167)



<br>

## Crédits

Au-delà des outils et bibliothèques utilisés, ce projet est rendu possible grâce aux projets open source suivants :
- [Leaflet.heat](https://github.com/Leaflet/Leaflet.heat) par [@jxn-30](https://github.com/jxn-30)
- [Leaflet.markercluster](https://github.com/Leaflet/Leaflet.markercluster) par [@eriknikulski](https://github.com/eriknikulski)
- [dom-to-image](https://github.com/tsayen/dom-to-image) par [@tsayen](https://github.com/tsayen)
