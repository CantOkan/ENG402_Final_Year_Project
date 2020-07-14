import os
import base64
import codecs

image = 'images/original_1_6.png'

encoded_string = ""
with open(image, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    file_code = encoded_string



file = codecs.open("image.txt", "w", "utf-8")
file.write(str(file_code))
file.close()