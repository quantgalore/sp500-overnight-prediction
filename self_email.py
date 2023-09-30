# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""
import smtplib

def send_message(message, subject):
    
    EMAIL = "yours@gmail.com"
    PASSWORD = "google-app-password, not your gmail password, but your application password"
    
    recipient = EMAIL
    auth = (EMAIL, PASSWORD)
 
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(auth[0], auth[1])
    
    subject = subject
    body = message
    message = f"Subject: {subject}\n\n{body}"
    server.sendmail(from_addr = auth[0], to_addrs = recipient, msg = message)
