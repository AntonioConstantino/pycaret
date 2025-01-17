import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Cargamos los datos
data = pd.read_csv(r'..\data\train_set.csv')
data = data.iloc[:, 1:]

# Filtramos y limpiamos los datos
data.dropna(inplace=True)

data = data[['start_station', 'end_lon', 'end_station', 'start_time',
             'duration', 'start_lat', 'trip_route_category', 'bike_id',
             'end_time', 'end_lat', 'start_lon', 'passholder_type']]

# Preparamos los datos
X = data.drop('passholder_type', axis=1)
y = data['passholder_type']

# Codificamos las variables categóricas
categorical_features = ['start_station', 'end_station', 'trip_route_category', 'bike_id', 'start_time', 'end_time']
numeric_features = ['end_lon', 'duration', 'start_lat', 'end_lat', 'start_lon']

# Pipelines de procesamiento de datos
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Creamos el pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),  # Reducimos las dimensiones con PCA
    ('classifier', RandomForestClassifier(random_state=123))
])

# Dividimos los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Definimos los parámetros para la búsqueda en GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth': [4, 5, 6, 7, 8],
    'classifier__criterion': ['gini', 'entropy']
}

# Optimización de hiperparámetros
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_

# Predicciones
y_pred = best_model.predict(X_test)

# Reporte de métricas
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar el mejor modelo
import joblib
joblib.dump(best_model, r'C:\Users\jacj2\OneDrive\Desktop\prueba_tecnica_Arkon\best_model.pkl')

# Cargar el modelo guardado
modelo_cargado = joblib.load(r'C:\Users\jacj2\OneDrive\Desktop\prueba_tecnica_Arkon\best_model.pkl')

# Predecir con el modelo cargado
predicciones = modelo_cargado.predict(X_test)
print(predicciones)

# Gráfica ROC
y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadero Positivo')
plt.title('Curva ROC')
plt.show()

print(f"Mejores hiperparámetros: {grid_search.best_params_}")
