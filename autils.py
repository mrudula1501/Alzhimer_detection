import tensorflow.keras
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1723728383<0 else 0
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('Alzhimer_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224,3), dtype=np.float32)

def predictAlzhimer():
	# Replace this with the path to your image
	image = Image.open('static/img/atest.jpg')

	#resize the image to a 224x224 with the same strategy as in TM2:
	#resizing the image to be at least 224x224 and then cropping from the center
	size = (224, 224)
	image = ImageOps.fit(image, size, Image.ANTIALIAS)

	#turn the image into a numpy array
	image_array = np.asarray(image)

	# display the resized image
	#image.show()

	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1   #(0 to 255  ==>> -1 to 1)

	# Load the image into the array
	data[0] = normalized_image_array

	# run the inference
	prediction = model.predict(data)
	print(prediction)
	idx = np.argmax(prediction)

	if idx == 0:
		return 'Mild Demented',"Mild Alzhimer Detected",'Exercising regularly. Exercising regularly can help to reduce the risk of developing Alzhimer disease'
	elif idx == 1:
		return 'Moderate Demented',"Moderate Alzhimer Detected",'Moderate Alzheimer’s is typically the longest stage and can last many years. People in the moderate stage of Alzheimer’s often require care and assistance.'
	elif idx == 2:
		return 'Non Demented',"Helathy MRI",'Exercising regularly. Exercising regularly can help to reduce the risk of developing Alzhimer disease.'


