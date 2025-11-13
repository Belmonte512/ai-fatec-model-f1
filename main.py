import os

import pandas as pd
import sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

import joblib
from sklearn.neural_network import MLPClassifier
from datetime import datetime


# -------------1. load dataset -------------
PATH_TO_CSV = "dataset/results_partial.csv"
df = pd.read_csv(PATH_TO_CSV)

races = pd.read_csv("dataset/races.csv")[['raceId', 'circuitId']]
df = df.merge(races, on='raceId', how='left')

# ------------- 2. criar target 'finished' -------------
df['finished'] = (df['statusId'] == 1).astype(int)

# ------------- 3. features (ajustadas ao SEU dataset) -------------
numerical_features = ['grid']
categorical_features = ['constructorId', 'driverId', 'circuitId']

df['constructorId'] = df['constructorId'].astype(str)
df['driverId'] = df['driverId'].astype(str)
df['circuitId'] = df['circuitId'].astype(str)

df['finished'].value_counts(normalize=True)


X = df[numerical_features + categorical_features].copy()
y = df['finished']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ------------- 4. Pipelines de pré-processamento -------------
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
    ('target', TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ]
)

pipe_mlp = Pipeline([
    ('pre', preprocessor),
    ('clf', MLPClassifier(max_iter=300, random_state=42))
])

grid_mlp = {
    'clf__hidden_layer_sizes': [(32,), (64,), (64,32)],
    'clf__activation': ['relu', 'tanh'],
    'clf__learning_rate_init': [0.001, 0.01]
}

# ------------- 5. Modelos + GridSearch -------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_log = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=5000))
])
grid_log = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__class_weight': [None, 'balanced']
}

pipe_rf = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_jobs=-1, random_state=42))
])
grid_rf = {
    'clf__n_estimators': [100, 300],
    'clf__max_depth': [None, 10, 20],
    'clf__class_weight': [None, 'balanced']
}

pipe_knn = Pipeline([
    ('pre', preprocessor),
    ('clf', KNeighborsClassifier())
])

grid_knn = {
    'clf__n_neighbors': [3, 5, 7, 11],
    'clf__weights': ['uniform', 'distance'],
    'clf__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

gs_log = GridSearchCV(pipe_log, grid_log, cv=cv, scoring='f1', n_jobs=-1)
gs_rf = GridSearchCV(pipe_rf, grid_rf, cv=cv, scoring='f1', n_jobs=-1)
gs_knn = GridSearchCV(pipe_knn, grid_knn, cv=cv, scoring='f1', n_jobs=-1)


print("Treinando Logistic Regression...")
gs_log.fit(X_train, y_train)

print("Treinando Random Forest...")
gs_rf.fit(X_train, y_train)

print("Treinando KNN...")
gs_knn.fit(X_train, y_train)


# ------------- 6. Avaliação -------------
models = {
    'LogisticRegression': gs_log.best_estimator_,
    'RandomForest': gs_rf.best_estimator_,
    'KNN': gs_knn.best_estimator_
}

# ------------- 9. avaliar no test set -------------
results = {}

print("\n\n================ VALIDAÇÃO TREINAMENTO vs TESTE ================")

for name, model in models.items():

    # --- Previsão no treino ---
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    train_f1 = f1_score(y_train, y_train_pred)
    train_acc = accuracy_score(y_train, y_train_pred)

    # --- Previsão no teste ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_f1 = f1_score(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)

    # --- GAP ---
    f1_gap = train_f1 - test_f1
    acc_gap = train_acc - test_acc

    print(f"\n### Modelo: {name}")
    print(f"Treino - F1: {train_f1:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Teste  - F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}")
    print(f"Gap F1: {f1_gap:.4f}   |   Gap ACC: {acc_gap:.4f}")

    # Diagnóstico automático
    if f1_gap < 0.05:
        print("Diagnóstico: 👍 Generalização excelente, sem overfitting.")
    elif f1_gap < 0.15:
        print("Diagnóstico: ⚠️ Um pouco de overfitting, aceitável.")
    else:
        print("Diagnóstico: ❌ Overfitting forte — modelo está decorando o treino.")


    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'confusion': cm
    }

    print(f"\n== {name} ==")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC_AUC: {roc:.4f}")
    print("Matriz de Confusão:\n", cm)

# ------------- 10. salvar melhor modelo -------------
best_name = max(results, key=lambda x: results[x]['f1'])
best_model = models[best_name]

os.makedirs("models", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_filename = f"models/best_model_{best_name}_{timestamp}.joblib"
joblib.dump(best_model, model_filename)

print("\n\n================ INTERPRETAÇÃO DETALHADA POR MODELO ================\n")

for name, model in models.items():
    print(f"\n### Modelo: {name}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    total_pos = tp + fn
    total_neg = tn + fp

    # evitar divisões por zero
    pos_acc = (tp / total_pos * 100) if total_pos > 0 else 0
    neg_acc = (tn / total_neg * 100) if total_neg > 0 else 0

    print(f"- Verdadeiros Positivos (TP): {tp}")
    print(f"- Falsos Negativos (FN): {fn}")
    print(f"- Verdadeiros Negativos (TN): {tn}")
    print(f"- Falsos Positivos (FP): {fp}")

    print(f"\nInterpretação:")
    print(f"O modelo classificou corretamente {tp} de {total_pos} casos onde o piloto **terminou** a corrida "
          f"({pos_acc:.1f}% de acerto nesse grupo).")
    print(f"Por outro lado, classificou corretamente {tn} de {total_neg} casos onde o piloto **não terminou** "
          f"({neg_acc:.1f}% de acerto nesse grupo).")

    print("Isso significa que:")
    print(f"- {tp} pilotos foram corretamente previstos como terminando.")
    print(f"- {tn} pilotos foram corretamente previstos como não terminando.")
    print(f"- {fn} pilotos terminaram, mas o modelo errou e disse que não terminariam.")
    print(f"- {fp} pilotos não terminaram, mas o modelo errou e disse que terminariam.\n")

# gerar timestamp no formato yyyy-mm-dd_hh-mm-ss
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# criar tabela resumo dos modelos
result_rows = []
for name, model in results.items():
    cm = model['confusion']
    tn, fp, fn, tp = cm.ravel()

    result_rows.append({
        'model': name,
        'accuracy': model['accuracy'],
        'precision': model['precision'],
        'recall': model['recall'],
        'f1': model['f1'],
        'roc_auc': model['roc_auc'],
        'true_positive': tp,
        'false_negative': fn,
        'true_negative': tn,
        'false_positive': fp
    })

results_df = pd.DataFrame(result_rows)

# arredondar todas as métricas numéricas para 4 casas decimais
results_df[['accuracy','precision','recall','f1','roc_auc']] = \
    results_df[['accuracy','precision','recall','f1','roc_auc']].round(4)

# salvar CSV com timestamp automático
os.makedirs("results", exist_ok=True)

# montar o caminho completo
filename = f"results/resultados_modelos_{timestamp}.csv"

# salvar o CSV
results_df.to_csv(filename, index=False)

print(f"\nResultados salvos em: {filename}")
