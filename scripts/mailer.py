"""
Script d'envoi automatique des modèles et de la documentation par email.
Utilise des valeurs en dur (NON RECOMMANDÉ, mais modifié pour Gmail).
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Mailer:
    def __init__(self, smtp_user, smtp_pass, recipients, subject, body):
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.recipients = recipients
        self.subject = subject
        self.body = body

    def send_email(self, attachment_path):
        msg = MIMEMultipart()
        msg['From'] = self.smtp_user
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = self.subject

        msg.attach(MIMEText(self.body, 'plain'))

        try:
            with open(attachment_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition',
                               f'attachment; filename={os.path.basename(attachment_path)}')
                msg.attach(part)
        except FileNotFoundError:
            logging.error(f"Fichier joint introuvable : {attachment_path}")
            return

        try:
            # Paramètres SMTP pour Gmail
            smtp_server = 'smtp.gmail.com'
            smtp_port = 587  

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  
            server.login(self.smtp_user, self.smtp_pass)
            server.send_message(msg)
            server.quit()  
            logging.info(f"Email envoyé avec succès à {', '.join(self.recipients)}!")
        except Exception as e:
            logging.error(f"Erreur lors de l'envoi de l'email : {e}")

if __name__ == "__main__":
    recipients = [
        "issabachir6@gmail.com",
    ]  # liste des emails
    sender = Mailer(
        smtp_user="issabachir6@gmail.com",
        smtp_pass="jdih zhts fvky fxai",
        recipients=recipients,
        subject="Modèle et Documentation",
        body="Veuillez trouver ci-joint le modèle et la documentation générés."
    )
    sender.send_email("output.zip")
