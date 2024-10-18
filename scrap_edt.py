import os
import openai
import requests
from ics import Calendar
import pytz
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")
# Fonction pour récupérer l'EDT depuis l'URL ICS
def get_edt(user_id):
    """Récupère l'emploi du temps (EDT) d'un utilisateur à partir d'une URL ICS.

    Cette fonction fait une requête GET pour obtenir le calendrier ICS d'un utilisateur
    via son identifiant. Elle extrait les événements du calendrier et les formate
    en une liste de dictionnaires contenant les détails des cours.

    Args:
        user_id (str): L'identifiant de l'utilisateur pour lequel récupérer l'emploi du temps.

    Raises:
        Exception: Si la requête échoue ou si l'identifiant est invalide.

    Returns:
        List[Dict[str, str]]: Une liste de dictionnaires, chaque dictionnaire représentant
        un cours avec les clés suivantes :
            - "nom_cours": Le nom du cours (str).
            - "début": La date et l'heure de début du cours au format 'YYYY-MM-DD HH:MM' (str).
            - "fin": La date et l'heure de fin du cours au format 'YYYY-MM-DD HH:MM' (str).
            - "description": La description du cours (str).
    """
    ics_url = f"http://applis.univ-nc.nc/cgi-bin/WebObjects/EdtWeb.woa/2/wa/default?login={user_id}%2Fical"
    response = requests.get(ics_url)

    # S'assurer que ça soit bien encoder en UTF-8
    response.encoding = "UTF-8"
    
    if response.ok:
        ics_content = response.text
    else:
        raise Exception("Veuillez entrer un identifiant valide 🚫")

    cal = Calendar(ics_content)
    cours = []

    local_tz = pytz.timezone("Pacific/Noumea")

    for event in cal.events:
        start_local = event.begin.astimezone(local_tz).strftime('%Y-%m-%d %H:%M')
        end_local = event.end.astimezone(local_tz).strftime('%Y-%m-%d %H:%M')

        cours.append({
            "nom_cours": event.name,
            "début": start_local,
            "fin": end_local,
            "description": event.description
        })

    return cours
