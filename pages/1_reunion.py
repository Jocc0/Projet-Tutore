import locale
import pytz
import streamlit as st
from langchain_openai import ChatOpenAI
from datetime import date, datetime, timedelta
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from streamlit_calendar import calendar
from scrap_edt import get_edt_semaine
from faiss_handler import transform_to_documents,save_to_faiss,retrieve_documents,retrive_documents_score,transform_weeks_to_documents
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
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

Ton rôle est d'offrir des conseils précis sur la gestion du temps, la répartition des matières, et les stratégies de révision efficaces. 
Tu dois poser des questions pour bien comprendre les objectifs de l'étudiant, ses priorités, et ses échéances. 
Tu es là pour l'accompagner dans ses révisions en proposant des suggestions d'amélioration et en lui fournissant des explications claires et adaptées à ses besoins.
**Présente le planning final sous forme de tableau pour faciliter la lecture.**
"""


PROMPT_TEMPLATE_REUNION="""
Voici les emplois du temps des deux utilisateurs avec leurs cours et leurs horaires.


Utilisateur 1 :
{edt1}

------

Utilisateur 2:
{edt2}

------

Crée un planning de disponibilités partagées pour permettre aux deux utilisateurs de trouver des créneaux libres pour se rencontrer en dehors de leurs heures de cours. 
Identifie les créneaux de disponibilité simultanée en tenant compte des horaires de cours de chacun et propose des créneaux adaptés pour des réunions de travail sur leur projet tutoré.
Donne aussi un bref sommaire des cours des deux utilisateurs

Assure-toi de respecter ces consignes :

Évite les chevauchements avec les heures de cours.
Propose des créneaux raisonnables en termes de durée, en priorisant des créneaux d'une à deux heures.
Le résultat attendu est un planning visuel ou un tableau des disponibilités communes entre les deux utilisateurs.
"""

def generate_planning_for_2(querry_text,main_user,second_user):
    try:
        # Utilisez model_name au lieu de model
        chat_model = ChatOpenAI(model_name="gpt-4o-mini")
        logging.info(f"Modèle initialisé")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation du modèle : {repr(e)}")
        return "Erreur lors de l'initialisation du modèle."


    #Création données 2ème utilisateur
    data_second_user=get_edt_semaine(second_user)
    docs=transform_weeks_to_documents(data_second_user,second_user)
    save_to_faiss(docs)
    

    # Récupère les documents pertinents à partir de FAISS
    edt_main_user = retrieve_documents(querry_text, main_user, 2)
    # Concatène tous les documents récupérés pour former le contexte
    context_main_user= "\n\n---\n\n".join([doc.page_content for doc in edt_main_user])
    logging.info(f"Document récupéré pour {main_user}: \n {context_main_user}")
    
    edt_second_user = retrieve_documents(querry_text,second_user,2)
    # Concatène tous les documents récupérés pour former le contexte
    context_second_user= "\n\n---\n\n".join([doc.page_content for doc in edt_second_user])
    logging.info(f"Document récupéré pour {second_user}: \n {context_main_user}")
    
    # Crée le prompt à partir du template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_REUNION)
    prompt = prompt_template.format(edt1=context_main_user,edt2=context_second_user)
    
    # Ajoute le message système pour guider le comportement de l'IA
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    logging.info(f"Message envoyé à l'IA:\n {messages}")
    # Utilise le modèle de chat pour générer la réponse
    try:
        response = chat_model(messages=messages)
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
date_debut = st.date_input("Sélectionner la date de début pour la réunion :", min_value=date.today())
# Calcul de la date de fin max (14 jours après la date de début)
date_fin_max = date_debut + timedelta(weeks=2)

date_fin = st.date_input("Sélectionner la date de fin :", min_value=date_debut, max_value=date_fin_max)
# Bouton pour valider les entrées
if st.button("Création du planning"):
    # Appel de la fonction avec les entrées
    resultat = generate_planning_for_2(querry_text,user_id1, user_id2)
    
    # Affichage du résultat
    st.write(resultat)


