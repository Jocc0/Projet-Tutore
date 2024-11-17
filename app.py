from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from streamlit_calendar import calendar
from scrap_edt import get_edt
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from tools import load_and_save_to_faiss_json,fetch_and_concatenate_documents
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

# Obtenez la date actuelle
current_date = datetime.now().strftime("%Y-%m-%d")


# System Prompt to define the assistant's role
SYSTEM_PROMPT = f"""
Aujourd'hui, nous sommes le {current_date}.

Tu es un assistant intelligent conçu pour aider un étudiant à organiser ses révisions et à créer un emploi du temps adapté. 
Ton rôle est d'offrir des conseils précis sur la gestion du temps, la répartition des matières, et les stratégies de révision efficaces. 
Tu dois poser des questions pour bien comprendre les objectifs de l'étudiant, ses priorités, et ses échéances. 
Tu es là pour l'accompagner dans ses révisions en proposant des suggestions d'amélioration et en lui fournissant des explications claires et adaptées à ses besoins.
"""

# Chat prompt template
PROMPT_TEMPLATE = """
Réponds à la question suivante en utilisant le contexte ci-dessous. Vérifie que les informations utilisées concordent avec les questions de l'humain. Si tu n'es pas sûr de la réponse, n'hésite pas à demander des précisions.

{context}

---

Réponds à la question en utilisant le contexte ci-dessus : {question}

Exemple :
Question : Quel est le début de mon cours de Mathématiques ?
Réponse : Le début de votre cours de Mathématiques est le 10 janvier.
"""
    

def generate_response(querry_text, user_id,list_of_dates):
    
    try:
        # Utilisez model_name au lieu de model
        chat_model = ChatOpenAI(model_name="gpt-4o-mini")
        logging.info(f"Modèle initialisé")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation du modèle : {repr(e)}")
        return "Erreur lors de l'initialisation du modèle."

    # Récupère les documents pertinents à partir de FAISS
    context=fetch_and_concatenate_documents(querry_text,user_id=user_id,list_of_dates=list_of_dates,top_k=25)
    st.write(context)
    # Crée le prompt à partir du template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=querry_text)
    
    # Ajoute le message système pour guider le comportement de l'IA
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    # Utilise le modèle de chat pour générer la réponse
    try:
        response = chat_model(messages=messages)
        return response.content  # Renvoie le contenu de la réponse générée
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse: {repr(e)}")
        return "Erreur lors de la génération de la réponse."
    
# Fonction principale pour la page web
def main():

    st.title("Chatbot🤖")

    user_id = st.text_input("Entrez votre identifiant: ", "")

    if 'edt' not in st.session_state:
        st.session_state.edt = None

    if st.button("Valider"):
        if user_id:
            try:
                cours = get_edt(user_id)
                st.session_state.edt = cours
                st.success("Identifiant validé ! Voici votre emploi du temps :")
                load_and_save_to_faiss_json()
                
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Veuillez entrer un identifiant valide.")

    # Définition des événements avec les couleurs pour TD, TP, CM
    if st.session_state.edt:
        events = []
        for cours in st.session_state.edt:
            
            color = ""
            if "TD" and "Td" in cours['nom_cours']:
                color = "#32a852"  # TD en vert clair
            elif "TP" and "Tp" in cours['nom_cours']:
                color = "#ffb22e"  # TP en orange clair
            elif "CM" and "Cm" in cours['nom_cours']:
                color = "#4287f5"  # CM en bleu clair
            else:
                color = "gray"  # Sinon par défaut en gris

            events.append({
                'start': cours['début'],
                'end': cours['fin'],
                'title': cours['nom_cours'],
                'description': cours['description'],
                'color': color
            })
    else:
        events = []

    # Affichage du calendrier dans un conteneur
    calendar(
    events=events,
    options={
        "locale": "fr",  # Tout en français
        'initialView': 'dayGridMonth',  # Vue par défaut : mois
        'editable': True,
        'headerToolbar': {
            'left': 'prev,today,next',
            'center': 'title',
            'right': 'dayGridMonth,timeGridWeek,timeGridDay'
        },
        "buttonText": {
            "today": "Aujourd'hui",  # Change le texte du bouton Today
            "month": "Mois",  # Change le texte du mode Mois
            "week": "Semaine",  # Change le texte du mode Semaine
            "day": "Jour"  # Change le texte du mode Jour
        },
        "slotMinTime": "07:00:00", #C'est l'heure minimale sur le calendrier
        "slotMaxTime": "19:00:00", #C'est l'heure maximale sur le calendrier
        "scrollTime": "07:00:00", # C'est l'heure à laquelle le calendrier en mode jour ou semaine commence
        "allDaySlot": False,  # permet ici de désactiver l'affichage des événements "All Day" (prennait de la place)
        "height": 'auto',  # La hauteur s'ajustera automatiquement ici *-*
        "contentHeight": 'auto',  # ça évite le défilement vertical dans le calendrier
        "expandRows": True,  # Prends/remplit l'espace disponible
        "eventMaxHeight": 20,  # Hauteur maximale des événements
        "stickyHeaderDates": True, # Permet de garder l'en-tête fixe, si l'utilisateur doit défiler son emploi du temps pendant le mode semaine/jour

    },
    custom_css="""
    .fc-event-past { opacity: 0.8; } #Change l'opacité des événements déjà passé
    .fc-event-time { font-style: italic; }
    .fc-event-title { font-weight: 700; font-size: 0.85rem; }  # Réduire la taille du texte des événements
    .fc-daygrid-event { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }  # Limiter la largeur des événements et ajouter des points de suspension
    .fc-daygrid-block-event { max-height: 20px; }  # Restreindre la hauteur des événements dans la vue "Mois"
    .fc-toolbar-title { font-size: 2rem; }
    .fc-timegrid-slot { height: auto !important; }  # Ajuste la hauteur des lignes dans la vue "semaine" et "jour"
    .fc-day-today {background: #f5f5f5 !important;}
    """
    )

    st.title("Chat :")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.write("Planifiez vos révisions (maximum 1 semaine) :")

    # Sélection de la date de début
    date_debut = st.date_input("Choisissez une date de début :")

    # Calcul de la date de fin maximale (7 jours après la date de début)
    date_fin_max = date_debut + timedelta(weeks=1)

    # Sélection de la date de fin
    date_fin = st.date_input(
        "Choisissez une date de fin (dans la semaine suivant la date de début) :",
        min_value=date_debut, 
        max_value=date_fin_max
    )
    # Générer une liste de dates entre date_debut et date_fin pour les utiliser dans la recherche
    if date_fin >= date_debut:  
        liste_dates = [
            (date_debut + timedelta(days=i)).isoformat()
            for i in range((date_fin - date_debut).days + 1)
        ]
    
    # React to user input
    if prompt := st.chat_input("Quand devrais-je réviser?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Here you would typically generate a response from your AI model
        response = generate_response(prompt,list_of_dates=liste_dates,user_id=user_id)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    main()
