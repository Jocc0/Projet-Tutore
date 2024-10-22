import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from streamlit_calendar import calendar
from scrap_edt import get_edt
from faiss_handler import transform_to_documents,save_to_faiss

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# Fonction pour générer une réponse via OpenAI
def generate_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content

def initialize_session_state():
    initialize_session_state()

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

    # Interaction avec le chatbot
    user_input = st.text_input("Vous: ", "")
    if st.button("Envoyer"):
        if user_input:
            st.write(f"Vous: {user_input}")
            bot_response = generate_response(user_input)
            st.write(f"Chatbot: {bot_response}")
        else:
            st.write("Veuillez entrer un message.")

def test():
    user_id="rcastelain"
    data=get_edt(user_id)
    docs=transform_to_documents(data,user_id)
    save_to_faiss(docs)

if __name__ == "__main__":
    test()
