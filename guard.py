import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from openai import OpenAI
from dotenv import load_dotenv
from guardrails import Guard
from guardrails.hub import QARelevanceLLMEval
from french_toxic_language_validator import FrenchToxicLanguage
from deep_translator import GoogleTranslator
import logging
import warnings
import re
# Désactiver les warnings inutiles
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'API OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Prompt principal du chatbot
META_PROMPT = """Tu es un assistant universitaire virtuel dédié à aider les étudiants dans leurs études et leur organisation universitaire.
Ton rôle est de fournir des conseils pour les révisions, de partager des méthodes d'apprentissage efficaces, de répondre aux questions sur les cours et les sujets académiques,
et d'organiser un emploi du temps optimisé pour leur permettre de mieux gérer leurs études et leur temps.

Tu as accès aux emplois du temps des étudiants (en format .ics transformé en texte), aux cours en format PDF,
et tu stockes ces informations en mémoire via une base RAG (utilisant FAISS et LangChain) pour éviter de reparcourir les données à chaque interaction.
Tu peux générer des plans de révision personnalisés en fonction des cours et des examens à venir, aider à planifier les tâches académiques, et fournir des conseils de gestion du temps.
De plus, tu peux comparer plusieurs emplois du temps pour trouver des créneaux de travail ou de révision.

Ton langage et ton comportement doivent toujours rester professionnels, respectueux, et adaptés à un contexte universitaire.
Tu restes strictement dans les limites des sujets académiques et de soutien étudiant. En cas d’interactions non appropriées, Guardrails AI modère tes réponses,
en veillant à éviter les biais, le langage inapproprié, et en assurant une protection des informations personnelles.

Tu réponds principalement en français mais peux interpréter des demandes en anglais si nécessaire.
Ton objectif est de fournir des conseils, guider les étudiants, une aide utile, respectueuse et précise aux étudiants pour qu’ils réussissent leurs études et tirent le meilleur parti de leur vie universitaire.
"""

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialisation des validateurs de Guardrails
guard_fr = Guard().use(FrenchToxicLanguage(on_fail="reask"))
guard_en = Guard().use_many(
    QARelevanceLLMEval(on_fail="reask")
)

# Fonction pour générer une réponse avec OpenAI
def generate_response(prompt):
    logging.info("Génération de la réponse...")
    try:
        if prompt:  # Vérifie que le prompt n'est pas vide
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": META_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4o-mini",
            )
            response = completion.choices[0].message.content
            logging.info("Réponse générée par le bot : %s", response)
            return response
        else:
            logging.error("Prompt vide.")
            return None
    except Exception as e:
        logging.error("Erreur lors de la génération de la réponse : %s", str(e))
        print("Erreur lors de la génération de la réponse.")
        return None

# Fonction pour traduire un texte
def translate_text(text, target_langue):
    try:
        if text and len(text) <= 5000:  # Vérifie que le texte n'est pas vide et ne dépasse pas 5000 caractères
            return GoogleTranslator(source='auto', target=target_langue).translate(text=text)
        else:
            logging.error("Texte vide ou dépasse la limite de 5000 caractères.")
            return None
    except Exception as e:
        logging.error("Erreur de traduction : %s", str(e))
        return None

# Fonction pour détecter les informations personnelles, y compris les numéros de téléphone d'ici (Nouvelle-Calédonie).
def contains_personally_identifiable_information(text):
    patterns = [
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Emails
        r"\(\+687\) \d{2}\.\d{2}\.\d{2}",  # Numéro formaté avec le code (+687)
        r"\b\d{2}\.\d{2}\.\d{2}\b",  # Numéro au format 54.06.54
        r"\b\d{6}\b",  # Numéro compact comme 540654
        r"\b\d{3} \d{3}\b",  # Numéro avec espace comme 541 153
        r"\b\d{3}\.\d{3}\b",  # Numéro au format 541.153
        r"\b\d{2}\ \d{2}\ \d{2}\b",  # Numéro au format 54 06 54
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

# Liste d'une partie des sujets valides
VALID_TOPICS = [
    "université", "cours", "matière", "contrôle continu", "CC", "examen", "travail", 
    "devoirs", "révision", "organisation", "emploi du temps", "rattrapage",
    "méthodes de travail", "gestion du temps"
]
# Liste des messages d'introduction ou neutres
VALID_GREETINGS = ["bonjour", "salut", "hello", "bonsoir", "hey", "oyy", "yo"]

#Fonction pour vérifier si le sujet de la phrase est valide ou non
def restrict_to_topic(text):
    """Valide si le texte contient un sujet académique valide ou une salutation."""
    # Vérifie les salutations
    for greeting in VALID_GREETINGS:
        if text.lower().startswith(greeting):
            return True
        
    #Vérifie les sujets valides
    for topic in VALID_TOPICS:
        if topic.lower() in text.lower():  # Vérifie si le sujet est mentionné dans le texte
            return True
        
    return False

#Fonction pour ajouter une réponse spécifique aux messages d'introduction
def handle_greetings(user_input):
    """Gère les salutations pour fournir une réponse personnalisée."""
    if user_input.lower() in VALID_GREETINGS:
        return "\n\nRéponse du bot : Bonjour ! Comment puis-je vous aider aujourd'hui ? Posez-moi une question sur vos cours, examens, révisions ou organisation universitaire.\n"
    return None

# Fonction principale pour valider un message utilisateur
def validate_input(user_input):
    # Étape 1 : Vérification des informations personnelles
    if contains_personally_identifiable_information(user_input):
        return {
            "valid": False,
            "reason": "Votre message contient des informations personnelles sensibles. Veuillez les retirer."
        }

    # Étape 2 : Restriction au sujet académique
    if not restrict_to_topic(user_input):
        return {
            "valid": False,
            "reason": "Votre message semble hors du cadre académique. Veuillez poser une question liée à vos cours, examens, révisions ou organisation universitaire."
        }

    # Étape 3 : Validation avec Guardrails en français
    result_fr = guard_fr.validate(user_input, metadata={'original_prompt': user_input})
    if not result_fr.validation_passed:
        return {
            "valid": False,
            "reason": "Votre message contient un contenu inapproprié ou hors cadre académique. Veuillez reformuler."
        }

    # Étape 4 : Validation en anglais pour des contextes multilingues
    result_en = guard_en.validate(user_input, metadata={'original_prompt': user_input})
    if not result_en.validation_passed:
        return {
            "valid": False,
            "reason": "Votre message semble hors contexte académique. Reformulez votre question pour rester dans ce cadre."
        }

    # Si tout est valide
    return {"valid": True, "reason": ""}

# Fonction pour modérer l'entrée utilisateur
def moderated_input():
    user_input = input("\n\nVotre message : ").strip()
    
    # Vérifie si le message est une salutation
    greeting_response = handle_greetings(user_input)
    if greeting_response:
        print(greeting_response)
        return None  # Ne continue pas vers une validation stricte pour les salutations
    
    validation_result = validate_input(user_input)

    if validation_result["valid"]:
        print("Message validé. Traitement en cours...")
        return user_input  # Retourne le message validé
    else:
        print(validation_result["reason"])
        return None  # Retourne None si la validation échoue

# Fonction principale pour gérer l'interaction utilisateur
def chatbot_interaction():
    while True:
        user_input = moderated_input()
        if user_input is None:
            continue  # Si le message est invalidé, redemander un nouveau message

        # Génération de la réponse si le message est validé
        bot_response = generate_response(user_input)
        if bot_response:
            print(f"\n\nRéponse du bot : {bot_response}\n")
        else:
            print("Erreur : Réponse du bot vide. Veuillez réessayer.")

# Point d'entrée du programme
if __name__ == "__main__":
    chatbot_interaction()