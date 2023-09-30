# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:38:28 2023

@author: Local User
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
