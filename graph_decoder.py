import base64
from io import BytesIO
from PIL import Image

base64_image_string = "base_string"  # Replace with actual Base64 string
image_data = base64.b64decode(base64_image_string)
image = Image.open(BytesIO(image_data))
image.save("sentiment_distribution.png")
image.show()  
