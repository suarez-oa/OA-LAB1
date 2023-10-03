from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

@app.get("/")
def index():
    return {'Proyecto Individual de Oa Johan Javier Suarez Merchan'}

#http://127.0.0.1:8000


df = pd.read_csv('game_item_review_nuevo.csv', low_memory=False)

@app.get('/')
def message():
    return 'Proyecto Individual Oa Johan Javier Suarez Merchan'


@app.get('/PlayTimeGenre/')
def PlayTimeGenre(genre: str) -> dict:
    genre = genre.capitalize()
    genre_df = df[df[genre] == 1]
    year_playtime_df = genre_df.groupby('year')['playtime_forever'].sum().reset_index()
    max_playtime_year = year_playtime_df.loc[year_playtime_df['playtime_forever'].idxmax(), 'year']
    return{'Genero': genre, 'Año de lanzamiento con mas horas jugadas para Genero:': int(max_playtime_year)}


@app.get('/UserForGenre/')
def UserForGenre(genre: str) -> dict:
    genre = genre.capitalize()
    genre_df = df[df[genre] == 1]
    max_playtime_user = genre_df.loc[genre_df['playtime_forever'].idxmax(), 'user_id']
    year_playtime_df = genre_df.groupby('year')['playtime_forever'].sum().reset_index()
    playtime_list = year_playtime_df.to_dict(orient='record')
    result = {
        'Usuario con mas horas jugadas para Genero' + genre: max_playtime_user,
        'Horas jugadas': playtime_list}
    return result



@app.get('/UsersRecommend/')
def UsersRecommend(year: int) -> dict:
    df_filtrado = df[(df['year']== year) & (df['recommend'] == True) & (df['sentiment_analysis'] == 2)]
    if df_filtrado.empty:
        return {'error': 'no encontrado'}
    df_ordenado = df_filtrado.sort_values(by='sentiment_analysis', ascending=False)
    top_3_resenas = df_ordenado.head(3)
    resultado = {
        'Puesto 1': top_3_resenas.iloc[0]['title'],
        'Puesto 2': top_3_resenas.iloc[1]['title'],
        'Puesto 3': top_3_resenas.iloc[2]['title'],
    }
    return resultado

@app.get('/UsersNotRecommend/')
def UsersNotRecommend(year: int)-> dict:
    df_filtrado = df[(df['year']== year) & (df['recommend'] == False) & (df['sentiment_analysis'] <=1)]
    if df_filtrado.empty:
        return {'error': 'no encontrado'}
    df_ordenado = df_filtrado.sort_values(by='sentiment_analysis', ascending=False)
    top_3_resenas = df_ordenado.head(3)
    resultado = {
        'Puesto 1': top_3_resenas.iloc[0]['title'],
        'Puesto 2': top_3_resenas.iloc[1]['title'],
        'Puesto 3': top_3_resenas.iloc[2]['title'],
    }
    return resultado

@app.get('/sentiment_analysis/')
def sentiment_analysis(year: int) -> dict:
    filtra_df = df[df['year']== year]
    sentiment = filtra_df['sentiment_analysis'].value_counts()
    result = {
        'Positivo': int(sentiment.get(2, 0)),
        'Neutral': int(sentiment.get(1, 0)),
        'Negativo': int(sentiment.get(0, 0))
    }
    return result






muestra = df.head(10000)
tfidf = TfidfVectorizer(stop_words='english')
muestra = muestra.fillna('')

tfidf_matri = tfidf.fit_transform(muestra['review'])
cosine_similarity = linear_kernel(tfidf_matri, tfidf_matri)



@app.get('/recomendacion_usuario/{id_juego}')
def recomendacion_juego(id_juego: int) -> dict:
    if id_juego not in muestra['id'].values:
        return {'mensaje': 'No existe el id del juego.'}
    titulo = muestra.loc[muestra['id']== id_juego, 'title'].iloc[0]
    idx = muestra[muestra['title'] == titulo].index[0]
    sim_cosine = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:6]]
    sim_juegos = muestra['title'].iloc[sim_ind].values.tolist()
   
    return {'juegos recomendados': list(sim_juegos)}


@app.get('/recomendacion_juego/{id_producto}')
def recomendacion_usuario(id_usuario: int)-> dict:
    if id_usuario not in muestra['steam_id'].values:
        return {'mensaje': 'No existe el id del usuario.'}
    generos = muestra.columns[2:17]
    filter_df = muestra[(muestra[generos].any(axis=1)) & (muestra['steam_id'] != id_usuario)]
    tdfid_matrix_filtered = tfidf.transform(filter_df['review'])
    cosine_similarity_filtered = linear_kernel(tdfid_matrix_filtered, tdfid_matrix_filtered)
    idx = muestra[muestra['steam_id'] == id_usuario].index[0]
    sim_cosine = list(enumerate(cosine_similarity_filtered[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:6]]
    sim_juegos = filter_df['title'].iloc[sim_ind].values.tolist()

    return {'juegos recomendados por el usuario': list(sim_juegos)}
