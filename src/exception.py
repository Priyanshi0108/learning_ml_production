import traceback
from src.logger import logging
class MLProjectError(Exception):
    def __init__(self,message:str,original_exception:Exception | None = None):
        self.message = message
        self.original_exception = original_exception

        if original_exception is not None:
            tb = traceback.extract_tb(
                original_exception.__traceback__
            )

            if tb:
                last_frame  = tb[-1]
                self.file_name = last_frame.filename
                self.line_number = last_frame.lineno

            else:
                self.file_name = None
                self.line_number = None    
        else:
            self.file_name = None
            self.line_number = None


        super().__init__(self.message)    


    def __str__(self):
        location = ""
        if self.file_name:
            location = (
                f" | file={self.file_name}"
                f" | line={self.line_number}"
            )
        return f"{self.message}{location}"


class DataIngestionError(MLProjectError):
    pass

class DataValidationError(MLProjectError):
    pass

class DataTransformationError(MLProjectError):
    pass

class ModelTrainingError(MLProjectError):
    pass
    