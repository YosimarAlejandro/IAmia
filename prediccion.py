import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify
import joblib

# 📌 Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["prueba"]
progreso_col = db["progresos"]
tareas_col = db["tareas"]

# 🧩 Extraer y combinar datos
progresos = list(progreso_col.find())
tareas = {str(t["_id"]): t for t in tareas_col.find()}

# 🔄 Preparar dataset con inferencia de dificultad_objetivo
data = []
dificultad_map = {"fácil": 0, "media": 1, "difícil": 2}
inv_map = {v: k for k, v in dificultad_map.items()}

for p in progresos:
    tarea = tareas.get(str(p["id_tarea"]))
    if tarea:
        dificultad_actual = tarea.get("dificultad", "media")
        dificultad_num = dificultad_map.get(dificultad_actual, 1)

        if "puntaje" not in p or "correcto" not in p:
            continue  # saltar si falta algo

        # 🧠 inferencia de siguiente dificultad
        if p["correcto"]:
            dificultad_objetivo = min(dificultad_num + 1, 2)
        else:
            dificultad_objetivo = max(dificultad_num - 1, 0)

        data.append({
            "puntaje": p["puntaje"],
            "correcto": int(p["correcto"]),
            "dificultad_actual": dificultad_num,
            "dificultad_objetivo": dificultad_objetivo
        })

df = pd.DataFrame(data)
df.dropna(inplace=True)  # por seguridad

# 📊 Entrenamiento
X = df[["puntaje", "correcto", "dificultad_actual"]]
y = df["dificultad_objetivo"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
joblib.dump(clf, "modelo_dificultad.pkl")

# 🌐 Servidor Flask
app = Flask(__name__)
modelo = joblib.load("modelo_dificultad.pkl")

@app.route("/predecir", methods=["POST"])
def predecir_api():
    data = request.get_json()
    if not all(k in data for k in ("puntaje", "correcto", "dificultad_actual")):
        return jsonify({"error": "Faltan campos requeridos (puntaje, correcto, dificultad_actual)"}), 400

    dificultad_map = {"fácil": 0, "media": 1, "difícil": 2}
    dificultad_actual_num = dificultad_map.get(data["dificultad_actual"], 1)

    entrada = [[
        data["puntaje"],
        1 if data["correcto"] else 0,
        dificultad_actual_num
    ]]
    pred = modelo.predict(entrada)[0]
    etiquetas = {0: "fácil", 1: "media", 2: "difícil"}
    return jsonify({"dificultad_sugerida": etiquetas[pred]})

if __name__ == "__main__":
    app.run(port=3000)
