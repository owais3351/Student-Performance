# exception.py module inside src package is used for exception handling
# code Implementation
import sys

# Define Exception Funcation--> store detailed info about the exception
def error_message_detail(error,error_detail : sys):
     _,_,exc_tb=error_detail.exc_info()
     file_name=exc_tb.tb_frame.f_code.co_filename
     error_message="Error occured in python script name[{0}] line number[{1}] error message[{2}]".format(file_name,exc_tb.tb_lineno,str(error))

     return error_message

# Define custom exception
# we can create custom exception by defining a new class
# Useful when we want to handle specific errors  in our application in more descriptive way

class CustomException(Exception):
     def __init__(self,error_message,error_detail : sys ):
          super().__init__(error_message) 
          self.error_message=error_message_detail(error_message,error_detail=error_detail)

     def __str__(self):
      return self.error_message
     

