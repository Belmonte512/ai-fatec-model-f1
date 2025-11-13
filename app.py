# app.py

import streamlit as st
from predict_service import FinishPredictor

# caminho do modelo salvo (ajuste para o seu arquivo real)
MODEL_PATH = "model.joblib"  # <-- troque aqui

@st.cache_resource
def load_predictor():
    return FinishPredictor(MODEL_PATH)

st.title("🏎️ Predição de Terminar Corrida de F1")
st.write("Modelo treinado com histórico de piloto, equipe e circuito.")

predictor = load_predictor()

st.sidebar.header("Parâmetros de entrada")

grid = st.sidebar.number_input("Posição de largada (grid)", min_value=0, max_value=22, value=10, step=1)
constructorId = st.sidebar.text_input("constructorId (ID da equipe)", "6")
driverId = st.sidebar.text_input("driverId (ID do piloto)", "1")
circuitId = st.sidebar.text_input("circuitId (ID do circuito)", "1")

if st.button("Prever"):
    result = predictor.predict_single(
        grid=grid,
        constructorId=constructorId,
        driverId=driverId,
        circuitId=circuitId
    )

    st.subheader("Resultado da predição")
    classe = "Vai terminar ✅" if result['predicted_class'] == 1 else "Não vai terminar ❌"
    st.write(f"**Classe prevista:** {classe}")
    st.write(f"Probabilidade de terminar: **{result['prob_finished']:.2%}**")
    st.write(f"Probabilidade de não terminar: **{result['prob_not_finished']:.2%}**")
