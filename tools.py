import logging
from faiss_handler import retrieve_documents


def filter_data_userId(list_of_dates,user_id)->dict:
    return {
        "date": list_of_dates,
        "user_id":user_id
    }

def fetch_and_concatenate_documents(query_text, list_of_dates, user_id, top_k=65):
    """
    Récupère les documents pertinents à partir de FAISS, les concatène et retourne le contexte.

    Args:
        query_text (str): Le texte de la requête.
        list_of_dates (list[str]): Liste des dates pour filtrer les documents.
        user_id (str): L'utilisateur pour lequel les documents sont récupérés.
        top_k (int): Nombre maximum de documents à récupérer.

    Returns:
        str: Contexte concaténé des documents récupérés.
    """
    
    try:
        # Récupération des documents pertinents
        edt = retrieve_documents(query_text, filter_data_userId(list_of_dates, user_id), user_id, top_k)
        
        if edt:
            for doc in edt:
                logging.info(f"Document récupéré pour {user_id}: \n {doc.page_content}")
            
            # Concaténation des documents pour former le contexte
            context = "\n\n---\n\n".join([doc.page_content for doc in edt])
            logging.info(f"Contexte final pour {user_id}: \n {context}")
            
            return context
        else:
            logging.info(f"Aucun document pertinent trouvé pour l'utilisateur {user_id}.")
            return ""
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des documents pour {user_id}: {repr(e)}")
        return ""
