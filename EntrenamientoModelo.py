import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib
import os

df = pd.read_csv('resultados.csv')

print("Primeras 5 filas:")
print(df.head())
print("\nDistribución de clases:")
print(df['Etiqueta'].value_counts())

# Separar características y etiquetas
X = df.iloc[:, :-1].values  # Todas las columnas excepto la última
y = df.iloc[:, -1].values   # Etiqueta

print(f"\nForma de los datos: {X.shape}")
print(f"Clases únicas: {np.unique(y)}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list = []
f1_list = []
auc_list = []
conf_matrices = []

print("\n Iniciando validación cruzada")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{skf.get_n_splits()}")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(
        X_train_scaled, 
        y_train,
        early_stopping_rounds=50,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    
    acc_list.append(acc)
    f1_list.append(f1)
    auc_list.append(auc)
    
    # Guardar matriz de confusión
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

# Resultados de validación cruzada
print("\nResumen de validación cruzada:")
print(f"Accuracy promedio: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
print(f"F1-score promedio: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
print(f"AUC-ROC promedio: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

# Entrenar modelo final con todos los datos
print("\nEntrenando modelo final con todos los datos...")
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)

final_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

final_model.fit(X_scaled, y)


model_dir = "modelo_texturas"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "modelo_final.xgb")
scaler_path = os.path.join(model_dir, "scaler.pkl")

final_model.save_model(model_path)
joblib.dump(final_scaler, scaler_path)

print("\nModelo y scaler guardados en:")
print(f"  - Modelo: {model_path}")
print(f"  - Scaler: {scaler_path}")


fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(final_model, ax=ax, importance_type='weight')
plt.title('Importancia de características')
plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'importancia_caracteristicas.png'))
plt.show()


avg_conf_matrix = np.mean(conf_matrices, axis=0)
print("\nMatriz de confusión promedio:")
print(avg_conf_matrix)


results = final_model.evals_result()
plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['mlogloss'], label='Entrenamiento')
plt.title('Curva de aprendizaje')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida logarítmica')
plt.legend()
plt.savefig(os.path.join(model_dir, 'curva_aprendizaje.png'))
plt.show()

print("\n¡Entrenamiento completado exitosamente!")