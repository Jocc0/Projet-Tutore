import locale
import shutil
import pytz
import streamlit as st
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from scrap_edt import get_edt_semaine_json
from faiss_handler import save_to_faiss,json_to_documents
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from tools import fetch_and_concatenate_documents


load_dotenv()


##################SETUP DES LOGS###################
# Ensure the logs directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # Create the directory if it doesn't exist


script_name = os.path.splitext(os.path.basename(__file__))[0]
log_file = os.path.join(log_dir, f"{script_name}.warn.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Écrire dans le fichier log
        logging.StreamHandler()  # Écrire dans la console (terminal)
    ]
)
##############################################
# Définir la langue en français
locale.setlocale(locale.LC_TIME, "fr_FR")

# Fuseau horaire de la Nouvelle-Calédonie
noumea_tz = pytz.timezone("Pacific/Noumea")
# Obtenir la date actuelle en Nouvelle-Calédonie
current_date_noumea = datetime.now(noumea_tz).strftime("%A %d %B %Y")

# Obtenez la date actuelle
current_date = datetime.now().strftime("%Y-%m-%d")


SYSTEM_PROMPT = f"""
Aujourd'hui, nous sommes le {current_date}({current_date_noumea}).

Tu es un assistant intelligent conçu pour aider un étudiant à organiser ses révisions et à créer un emploi du temps adapté.
Ton rôle est d'offrir des conseils précis sur la gestion du temps. 
**Présente le planning final sous forme de tableau pour faciliter la lecture.**
"""


PROMPT_TEMPLATE_REUNION="""
Voici les emplois du temps des deux utilisateurs avec leurs cours et leurs horaires.
Les utilisateurs souhaitent organiser des réunions entre le {start_date} et le {end_date}.


Utilisateur 1 :
{edt1}

------

Utilisateur 2:
{edt2}

------
Objectif :
Le but est de trouver des créneaux de réunion où les deux utilisateurs sont disponibles en dehors de leurs horaires de cours. Les créneaux proposés doivent être d’une durée comprise entre 1 à 2 heures.

Méthodologie :
Éviter les chevauchements avec les horaires de cours.
Proposer des créneaux de réunion en tenant compte des périodes disponibles de chaque utilisateur. Par exemple, si l'Utilisateur 1 a un créneau libre de 10:00 à 14:00, et l'Utilisateur 2 a un créneau libre de 11:00 à 15:00, le créneau commun disponible est de 11:00 à 14:00.
Respecter les contraintes : Assurer que la durée des créneaux soit d’au moins 1 heure et au maximum 2 heures.
Organiser les créneaux par jour de la semaine et proposer les meilleurs moments pour une réunion.
Le résultat attendu est un planning visuel ou un tableau des disponibilités communes entre les deux utilisateurs.
"""

def load_and_save_to_faiss_json(user_id):
    get_edt_semaine_json(user_id)
    docs=json_to_documents(user_id)
    save_to_faiss(docs)


def generate_planning_for_2(querry_text,main_user,second_user,list_of_dates):
    try:
        # Utilisez model_name au lieu de model
        chat_model = ChatOpenAI(model_name="gpt-4o-mini")
        logging.info(f"Modèle initialisé")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation du modèle : {repr(e)}")
        return "Erreur lors de l'initialisation du modèle."

    #On supprime parce que j'arrive pas trop à gérer les doublons for now
    st.write("Génération des fichiers ...")
    folder="faiss_data"
    if os.path.exists(folder) and os.path.isdir(folder):
        try:
            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                # Check if it is a file before deleting
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                # Optionally handle subfolders (comment out if not needed)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Removed folder: {file_path}")
        except Exception as e:
            print(f"An error occurred while clearing the folder: {e}")
    else:
        print(f"The folder '{folder}' does not exist.")

    #Création données pour les deux utilisateurs avec l'embeding et tout le tralala
    st.write(f"Génération pour {main_user}")
    load_and_save_to_faiss_json(main_user)
    st.write(f"Génération pour {second_user}")
    load_and_save_to_faiss_json(second_user)

    st.write("Création du contexte ...")
    #On récupère les données et on les manipules pour récupérer tout en une chaine de caractere 
    context_main_user=fetch_and_concatenate_documents(querry_text,list_of_dates,main_user)
    context_second_user=fetch_and_concatenate_documents(querry_text,list_of_dates,second_user)
    
    # Crée le prompt à partir du template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_REUNION)
    prompt = prompt_template.format(edt1=context_main_user,edt2=context_second_user,start_date=list_of_dates[0],end_date=list_of_dates[len(list_of_dates)-1])
    
    # Ajoute le message système pour guider le comportement de l'IA
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    logging.info(f"Message envoyé à l'IA:\n {messages}")
    # Utilise le modèle de chat pour générer la réponse
    st.write("Génération de la réponse")
    try:
        st.write(messages)
        response = chat_model(messages=messages)
        status.update(label="Download complete!", state="complete", expanded=False )
        return response.content  # Renvoie le contenu de la réponse générée
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse: {repr(e)}")
        return "Erreur lors de la génération de la réponse."

st.set_page_config(page_title="Réunion", page_icon="📅")

st.markdown("# Réunion")
st.write(
    """Création de réunion avec d'autres utilisateurs"""
)



user_id1 = st.text_input("Entrez votre identifiant: ", "rcastelain")
user_id2 = st.text_input("Entrez deuxième identifiant: ", "htiaiba")
# Sélection des dates
date_debut = st.date_input("Sélectionner la date de début pour la réunion :")
#Pour l'instant on met pas vu qu'on a plus de cours
#date_debut = st.date_input("Sélectionner la date de début pour la réunion :", min_value=date.today())
# Calcul de la date de fin max (14 jours après la date de début)
date_fin_max = date_debut + timedelta(weeks=2)

date_fin = st.date_input("Sélectionner la date de fin :", min_value=date_debut, max_value=date_fin_max)

# Générer une liste de dates entre date_debut et date_fin pour les utiliser dans la recherche
if date_fin >= date_debut:
    liste_dates = [
        (date_debut + timedelta(days=i)).isoformat()
        for i in range((date_fin - date_debut).days + 1)
    ]
    
# Bouton pour valider les entrées
if st.button("Création du planning"):
    # Appel de la fonction avec les entrées
    with st.status("Génération de la réponse...", expanded=True) as status:
        resultat = generate_planning_for_2(f"Cours entre date_debut : {date_debut}  date_fin : {date_fin}",user_id1, user_id2, liste_dates)

    # Affichage du résultat
    st.write(resultat)


