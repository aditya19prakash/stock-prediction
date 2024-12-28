import socket
import logging
import os

# Ensure the log directory exists
log_directory = 'log'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(filename=os.path.join(log_directory, 'error.log'), level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def check_internet_connection():
    try:
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError as e:
        logging.error(f"Error in check_internet_connection: {e}")
        return False

def format_number(num):
    try:
        s = str(int(num))
        if len(s) <= 3:
            return s
        result = s[-3:]
        s = s[:-3]    
        while s:
            result = s[-2:] + ',' + result if s[-2:] else s[-1] + ',' + result
            s = s[:-2] 
        return result
    except Exception as e:
        logging.error(f"Error in format_number: {e}")
        return None