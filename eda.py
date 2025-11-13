# ============================================================
#                    EDA COMPLETO by ChatGPT
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações gráficas
plt.style.use("ggplot")
sns.set(font_scale=1.1)

# ===================== 1. leitura dos dados =====================

df = pd.read_csv("dataset/results_partial.csv")

races = pd.read_csv("dataset/races.csv")[['raceId', 'circuitId']]
df = df.merge(races, on='raceId', how='left')

# criar target
df['finished'] = (df['statusId'] == 1).astype(int)

# converter categorias
df['constructorId'] = df['constructorId'].astype(str)
df['driverId'] = df['driverId'].astype(str)
df['circuitId'] = df['circuitId'].astype(str)

numerical_features = ['grid', 'laps', 'points', 'positionOrder']
categorical_features = ['constructorId', 'driverId', 'circuitId']

# ===================== 2. informações gerais =====================

print("\n===== INFO DO DATASET =====")
print(df.info())

print("\n===== PRIMEIRAS 10 LINHAS =====")
print(df.head(10))

print("\n===== ESTATÍSTICAS DESCRITIVAS =====")
print(df[numerical_features].describe())

# ===================== 3. distribuição do target =====================

print("\n===== DISTRIBUIÇÃO DO TARGET (finished) =====")
print(df['finished'].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
sns.countplot(x='finished', data=df, palette='Set2')
plt.title("Distribuição do Target: Finished")
plt.show()

# ===================== 4. distribuição das features numéricas =====================

for col in numerical_features:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Histograma de {col}")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")

    plt.tight_layout()
    plt.show()

# ===================== 5. distribuição das features categóricas =====================

for col in categorical_features:
    plt.figure(figsize=(12, 4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index[:20])
    plt.title(f"Top 20 categorias mais frequentes - {col}")
    plt.xticks(rotation=45)
    plt.show()

# ===================== 6. correlação =====================

plt.figure(figsize=(10, 6))
corr = df[numerical_features + ['finished']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Heatmap de Correlação")
plt.show()

# ===================== 7. scatterplots úteis =====================

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='grid', y='positionOrder', hue='finished')
plt.title("Grid vs Posição Final")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='laps', y='points', hue='finished')
plt.title("Laps vs Points")
plt.show()

# ===================== 8. análise de outliers =====================

for col in numerical_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\n===== OUTLIERS EM {col} =====")
    print(f"Total: {len(outliers)}")
    print(outliers[[col]].head())

# ===================== 9. insights automáticos =====================

print("\n================ INSIGHTS AUTOMÁTICOS ================\n")

# Target balance
pos = df['finished'].mean() * 100
if pos < 30:
    print(f"- O dataset é desbalanceado: apenas {pos:.1f}% de pilotos terminaram.")
elif pos > 70:
    print(f"- O dataset é desbalanceado: {pos:.1f}% de pilotos terminaram.")
else:
    print(f"- O target é relativamente balanceado: {pos:.1f}% terminam a corrida.")

# Correlações relevantes
high_corr = corr['finished'].abs().sort_values(ascending=False)
print("\nCorrelação com o target:")
print(high_corr)

print("\n======================================================")
print("EDA COMPLETA GERADA COM SUCESSO 🎉")
print("======================================================")
