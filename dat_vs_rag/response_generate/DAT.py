
from dat_vs_rag.chroma_db.BM25 import get_BM25_scores
from dat_vs_rag.chroma_db.ModernBert import semantic_scores

from agents.models import Arcee_Mini


def generate_grades(query: str, top1_lex: str, top1_sem: str) ->dict[str, int]:

    '''
    получаем оценки релевантности лексического и семантического поисков
    '''

    str_grades = Arcee_Mini(
       f"""
        You are an evaluator assessing the retrieval effectiveness of dense retrieval 
        ( Cosine Distance ) and BM25 retrieval for finding the correct answer .

        ## Task :
        Given a question and two top1 search results ( one from dense retrieval ,
        one from BM25 retrieval ) , score each retrieval method from **0 to 5**
        based on whether the correct answer is likely to appear in top2 , top3 , etc .

        ### ** Scoring Criteria :**
        1. ** Direct hit --> 5 points **
        - If the retrieved document directly answers the question , assign **5 points **.

        2. ** Good wrong result ( High likelihood correct answer is nearby ) --> 3 -4 points **
        - If the top1 result is ** conceptually close ** to the correct answer ( e . g . , mentions relevant entities ,
        related events , partial answer ) , it indicates the search method is in the right direction .
        - Give **4** if it 's very close , **3** if somewhat close .

        3. ** Bad wrong result ( Low likelihood correct answer is nearby ) --> 1 -2 points **
        - If the top1 result is ** loosely related but misleading ** ( e . g . , shares keywords but changes context ) ,
        correct answers might not be in top2 , top3 .
        - Give **2** if there 's a small chance correct answers are nearby , **1** if unlikely .

        4. ** Completely off - track --> 0 points **
        - If the result is ** totally unrelated ** , it means the retrieval
        method is failing .

        ---
        ### ** Given Data :**
        - ** Question :** "{ query }”
        - ** dense retrieval Top1 Result :** "{ top1_sem }"
        - ** BM25 retrieval Top1 Result :** "{ top1_lex }"
        ---

        ### ** Output Format :**
        Return two integers separated by a space :
        - ** First number :** dense retrieval score .
        - ** Second number :** BM25 retrieval score .
        - Example output : 3 4
        ( Vector : 3 , BM25 : 4)
        ** Do not output any other text .**
        """
    )

    grades = str_grades.split(" ")

    return {
        "sem": int(grades[0]),
        "lex": int(grades[1])
    }

def calculate_alpha(grades: dict[str, int]) ->float:

    '''
    функция расчета коэффициента альфа из оценок релевантности лексического и семантического поисков
    '''

    if grades["sem"]==0 and grades["lex"]==0:
        return 0.5
    elif grades["sem"]==5 and grades["lex"]!=5:
        return 1.0
    elif grades["lex"]==5 and grades["sem"]!=5:
        return 0.0
    else:
        return grades["sem"]/(grades["sem"] + grades["lex"])
    

def generate_alpha_coef(query:str, lex_scores: dict[str, float], sem_scores: dict[str, float]) ->float:

    '''
    генерирует тот самый альфа коэффициент
    '''

    max_lex_score = max(lex_scores.values())
    top1_lex = next((doc for doc in lex_scores if lex_scores[doc] == max_lex_score))

    max_sem_score = max(sem_scores.values())
    top1_sem = next((doc for doc in sem_scores if sem_scores[doc] == max_sem_score))

    grades = generate_grades(query, top1_lex, top1_sem)
    alpha = calculate_alpha(grades)

    return alpha


def get_hibrid_scores(query: str) ->dict:

    '''
    возвращает гибридный скор между запросом и каждым докуметом
    '''

    BM25_scores = get_BM25_scores(query)
    sem_scores = semantic_scores(query)

    hibrid_scores = {}

    alpha = generate_alpha_coef(query, BM25_scores, sem_scores)

    for document in BM25_scores:
        hibrid_scores[document] = alpha * sem_scores[document] + (1-alpha) * BM25_scores[document]

    return hibrid_scores

def get_top3_docs(scores: dict[str, float]) ->list[str]:

    '''
    ищет топ 3 в словаре со всеми документами и их скорами 
    '''

    top3 = [{"doc1": -10.0}, {"doc2": -10.0}, {"doc3": -10.0}]

    for doc in scores:
        score = scores[doc]

        if score > list(top3[2].values())[0]:
            if score > list(top3[1].values())[0]:
                if score > list(top3[0].values())[0]:
                    top3 = [{doc: score}] + top3[:2]
                else:
                    top3[2] = top3[1]
                    top3[1] = {doc: score}
            else:
                top3[2] = {doc: score}
    
    return [list(top.keys())[0] for top in top3]

def get_DAT_context(query: str) ->list[str]:
    '''
    возвращает контекст по запросу
    '''

    docs_with_hibrid_score = get_hibrid_scores(query)

    top3 = get_top3_docs(docs_with_hibrid_score)

    return top3

