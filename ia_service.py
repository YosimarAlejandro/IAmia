from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np

# Crear la aplicación FastAPI
app = FastAPI()

# Cargar el modelo de TensorFlow
model = tf.keras.models.load_model(r"C:\Users\yosim\Desktop\ia_service\models\modelo_entrenado.h5")

# Modelo de entrada para recibir datos
class NotificationRequest(BaseModel):
    input_data: list  # Datos para la inferencia

# Función para enviar notificaciones por email
def send_email(to_email: str, subject: str, message: str):
    from_email = "al222210667@gmail.com"  # Cambia esto por tu email
    from_password = "kzjs cpwz zzpb bhqh"  # Usa la contraseña generada si tienes 2FA activado

    # Configurar el servidor SMTP (para Gmail en este caso)
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(from_email, from_password)

    # Crear el mensaje
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    # Asegurarse de que el mensaje se adjunte como UTF-8
    msg.attach(MIMEText(message, "plain", "utf-8"))  # Aquí es donde forzamos la codificación UTF-8

    # Enviar el email
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

# Endpoint para recibir datos y predecir si se debe enviar una notificación
@app.post("/predict_and_notify")
async def predict_and_notify(notification: NotificationRequest):
    # Preparar los datos de entrada para la inferencia
    input_data = np.array(notification.input_data).reshape(1, -1)  # Asegúrate de que el formato coincida con el modelo

    # Realizar la inferencia con el modelo
    prediction = model.predict(input_data)

    # Basado en la predicción, decidir si se debe enviar una notificación
    if prediction[0] > 0.5:  # Si la predicción es mayor que 0.5, se envía una notificación (por ejemplo)
        try:
            # Asegúrate de que el mensaje también esté en UTF-8
            message = "Tu predicción ha sido exitosa."
            send_email("destination_email@example.com", "Notificación importante", message)
            return {"status": "success", "message": "Notification sent successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}
    else:
        return {"status": "no_notification", "message": "No notification needed based on the prediction."}
