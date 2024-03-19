# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1:
#### Import numpy module as np and pandas as pd.
<br>

### Step 2:
#### Assign the values to variables in the program.
<br>

### Step 3:
#### Get the values from the user appropriately.
<br>

### Step 4:
#### Continue the program by implementing the codes of required topics.
<br>

### Step 5:
#### Thus the program is executed in google colab.
<br>

## Program:

### Developed By : SANJAY T

### Register Number : 212222110039

#### Installing OpenCV , importing necessary libraries and displaying images  

```py
# Install OpenCV library
!pip install opencv-python-headless

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images 
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


```
#### (i) Image Translation
```py
# Load an image from URL or file path
image_url = 'dip031.jpeg'  
image = cv2.imread(image_url)

# Define translation matrix
tx = 50  # Translation along x-axis
ty = 30  # Translation along y-axis
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])  # Create translation matrix

# Apply translation to the image
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

# Display original and translated images
print("Original Image:")
show_image(image)
print("Translated Image:")
show_image(translated_image)
```

#### (ii) Image Scaling
```py

# Load an image from URL or file path
image_url = 'dip032.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)


# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis


# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
show_image(image)
print("Scaled Image:")
show_image(scaled_image)

```




#### (iii) Image shearing
```py
# Load an image from URL or file path
image_url = 'dip033.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)

```



#### (iv) Image Reflection

```py
# Load an image from URL or file path
image_url = 'dip034.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)

# Reflect the image vertically
reflected_image_vertical = cv2.flip(image, 0)

# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)


```
##### (a) → Reflecting Horizontally

```py
# Display original and reflected images

show_image(image)
print("↑ Original Image")
show_image(reflected_image_horizontal)
print("↑ Reflected Horizontally")
```

##### (b) → Reflected Vertically

```py
show_image(image)
print("↑ Original Image")
show_image(reflected_image_vertical)
print("↑ Reflected Vertically")


```
##### (c) → Reflecting Horizontally & Vertically
```py

show_image(image)
print("↑ Original Image")
show_image(reflected_image_both)
print("↑ Reflected Both")

```

### (v) Image Rotation

```py
# Load an image from URL or file path
image_url = 'dip035.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)


# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)


```


### (vi) Image Cropping

```py
# Load an image from URL or file path
image_url = 'dip036.jpeg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)


```



## Output:

### (i) Image Translation
<br>
<br>

![11](https://github.com/sanjaythiyagarajan/IMAGE-TRANSFORMATIONS/assets/119409242/5dde52da-e2de-4c87-8358-8430c96d654d)





<br>
<br>

### (ii) Image Scaling
<br>
<br>

![22](https://github.com/sanjaythiyagarajan/IMAGE-TRANSFORMATIONS/assets/119409242/e402b2dc-9679-4797-946d-b912c75a22dc)





<br>
<br>


### (iii) Image shearing
<br>
<br>

![33](https://github.com/sanjaythiyagarajan/IMAGE-TRANSFORMATIONS/assets/119409242/5adf34da-e80b-4547-b3e6-ed23e3bb2b8a)




<br>
<br>


### (iv) Image Reflection


#### Reflecting Horizontally

![44](https://github.com/sanjaythiyagarajan/IMAGE-TRANSFORMATIONS/assets/119409242/745f6fef-a2f2-4f27-9463-237e8e2ccda5)





#### Reflecting Vertically

![55](https://github.com/sanjaythiyagarajan/IMAGE-TRANSFORMATIONS/assets/119409242/4e7f2405-da64-4882-b89f-39c601a5c1da)





#### Reflecting Horizontally & Vertically


![66](https://github.com/sanjaythiyagarajan/IMAGE-TRANSFORMATIONS/assets/119409242/9f152ceb-28fe-4ab3-af1d-53dd4364bd2b)


<br>
<br>


### (v) Image Rotation
<br>
<br>

![77](https://github.com/sanjaythiyagarajan/IMAGE-TRANSFORMATIONS/assets/119409242/583bd299-1955-4e98-9bc2-df15468a49c0)





<br>
<br>



### (vi) Image Cropping
<br>
<br>

![88](https://github.com/sanjaythiyagarajan/IMAGE-TRANSFORMATIONS/assets/119409242/8d14f98e-96ac-4b14-8b9e-31f33ae127d5)



<br>
<br>




## Result: 

### Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
