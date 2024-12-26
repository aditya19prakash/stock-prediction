import socket
def check_internet_connection():
    try:
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        return False
def format_number(num):
     s = str(int(num))
     if len(s) <= 3:
        return s
     result = s[-3:]
     s = s[:-3]    
     while s:
        result = s[-2:] + ',' + result if s[-2:] else s[-1] + ',' + result
        s = s[:-2] 
     return result