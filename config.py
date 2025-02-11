import os

class Config:
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {"csv"}
    SECRET_KEY = "supersecretkey"  # Change for production

    @staticmethod
    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
