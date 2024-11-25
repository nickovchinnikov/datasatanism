---
title: Dive into Learning from Data - MNIST Video Adventure
description: We're diving into the realm of MNIST, a dataset that's like a treasure map for budding data scientists.
authors:
  - nick
date:
  created: 2024-11-25
comments: true
categories:
  - Classification
  - Dimensionality Reduction
  - Feature Engineering
  - Data Visualization
tags:
  - Classification
  - PCA (Principal Component Analysis)
  - Logistic Regression
  - Machine Learning
  - MNIST
  - Python
  - Data Preprocessing
  - Feature Engineering
  - sklearn
  - numpy
---


![Image title](../assets/dive_into_learning_from_data/mnist_feat_eng_plot.png){ align=center, width="500" }

Hey there, data enthusiasts! Today, we're diving into the fascinating world of **Machine Learning Classification** using one of the most iconic datasets out there - the **MNIST dataset**. *MNIST stands for Modified National Institute of Standards and Technology.*

We're diving into the realm of MNIST - a dataset that's like a treasure map for budding data scientists. It contains thousands of handwritten digits from 0 to 9. Each image is a snapshot of someone's attempt to scribble a number, and our mission is to make sense of these scribbles.

<!-- more -->

<iframe width="1707" height="765" src="https://www.youtube.com/embed/csZ4dIAPowA" title="Dive Into Learning From Data" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


### What's Classification?


Classification is like teaching your computer to distinguish between cats and dogs in photos. You feed it images, labeled "cat" or "dog", and it learns to predict which label to slap on a new photo.

![Image title](../assets/dive_into_learning_from_data/cat_dog.png){ align=center, width="500" }
/// caption
Cat VS Dog
///


### Let's Meet MNIST


**MNIST** stands for **Modified National Institute of Standards and Technology** database. It's not about cats and dogs, but it's just as exciting. This dataset contains images of handwritten digits (0 through 9). Your mission, should you choose to accept it, is to train a model to identify these digits correctly.

![Image title](../assets/dive_into_learning_from_data/MNIST_more.png){ align=center }
/// caption
MNIST digits
///


### Setup and Data Exploration


First, you'll need to set up your environment. I'm rocking a Jupyter Notebook inside VS Code, but any Python environment will do the trick. Let's get our hands dirty with some code:

```python
# Import necessary libraries
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# Check what we've got in our dataset
print(mnist.keys())

# Separate the pixel information (images) and labels
X, y = mnist['data'], mnist['target']

# Let's peek at the data
print(X.shape)  # Should print (70000, 784) - 70k images, each 784 pixels
```


### Understanding Pixel Data


Notice how most of the data in `X` is close to zero? That's because most of the image is empty space:

```python
# Check the range of pixel intensities
print("Minimum pixel intensity:", X.min().min())
print("Maximum pixel intensity:", X.max().max())
```


### Data Insights:


- **Why 784 pixels?** Each image in MNIST is 28x28 pixels, which totals to 784 when flattened.
- **Pixel Intensity:** A value of 0 means the pixel is as black as a shadow, while 255 is a light as bright as the sun. In between, there's a spectrum of grays.
- **Data Dimensionality:** With 70,000 images, each with 784 features, we're dealing with a lot of information. But how much of it is truly useful?

![Image title](../assets/dive_into_learning_from_data/Black_hole.jpg){ align=center, width="500" }
/// caption
Direct radio image of a supermassive black hole at the core of Messier 87
///


### Focusing on the Relevant Data


Most of these images are like the night sky, mostly dark with occasional stars (activated pixels). Here's how we peek at the center of these images:

```python
# Look at the middle band of the images where digits usually reside
X.iloc[:, 400:500]
```


#### Questions to Explore:


- How does a computer learn from pixels what humans recognize by patterns?
- What patterns do we humans overlook that machines might find fascinating?


### Visualizing The Data


Images are typically represented as matrices where each cell might represent the intensity of a pixel. Here's how we can look at what we're dealing with:

```python
# Display the first image
some_digit = X.iloc[0].values.reshape(28, 28)
plt.imshow(some_digit, cmap='binary')
plt.axis("off")
plt.show()

print("Label for this image:", y[0])
```

This code reshapes the `X` data into a 28x28 matrix (since 28 * 28 = 784), which is the size of our digit images. We use `matplotlib` to visualize it.


### A Gallery of Digits


Alright, let's take a moment to appreciate the art of handwritten numbers! This snippet of code takes the first ten images from our dataset and lays them out in a grid.

```python
import matplotlib.pyplot as plt

# Setting up a 2x5 grid of subplots
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

# Loop through our subplots
for i, ax in enumerate(axes.flat):
    # Display the image in the subplot
    ax.imshow(X.iloc[i].values.reshape(28, 28), cmap="gray")
    # Set the title of each subplot to the digit label
    ax.set_title(y[i])
    # Turn off the axis ticks
    ax.axis("off")

# Adjust the layout to prevent overlapping
plt.tight_layout()
# Show the gallery
plt.show()
```

![Image title](../assets/dive_into_learning_from_data/mnist_grid.png){ align=center, width="500" }
/// caption
MNIST grid example
///

You've turned raw pixel data into a visual representation that's not only informative but also engaging, allowing us to see the variety in how digits are handwritten.


### Understanding Logistic Regression


Once we've got our dataset ready, it's time to apply a classification algorithm. We'll use **logistic regression**, which, despite its name, is designed for classification. It's not about predicting a continuous outcome; instead, it helps us decide whether an image represents a specific digit or not.

Logistic regression works by transforming input values through a **sigmoid function**, which squeezes any real number into a range between 0 and 1, effectively representing a probability. Here's how it looks:

- If the output of the sigmoid function is less than a certain threshold (commonly 0.5), we classify the input as belonging to one class.
- If it's greater than or equal to that threshold, we classify it into another class.

**What is the Sigmoid Function?**
The sigmoid function takes any real-valued number and squashes it into a range from 0 to 1. It's defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

#### The Sigmoid Function

Let's dive into the sigmoid function itself. Here's how you can define and visualize it:

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Compute the sigmoid of x.
    
    Parameters:
    x (float or numpy array): Input value or array of values.
    
    Returns:
    float or numpy array: The sigmoid output.
    """
    return 1 / (1 + np.exp(-x))

# Generate a range of x values
y = np.linspace(-10, 10, 100)

# Apply sigmoid function to generate y values
x = sigmoid(y)

# Plot the sigmoid function
plt.figure(figsize=(8, 6))
plt.plot(y, x, label=r"$\sigma(x) = \frac{1}{1 + e^{-x}}$", color='blue')
plt.axhline(y=0.5, linestyle="--", color="grey", label='Threshold = 0.5')  # Adding threshold line

plt.title("Sigmoid Function: The Heart of Logistic Regression")
plt.xlabel("Input (x)")
plt.ylabel("Output (Probability)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Remember, this isn't just about plotting a curve;** it's about understanding how logistic regression decides on class boundaries using the sigmoid function to transform our input into a probability of belonging to a certain class.

![Image title](../assets/dive_into_learning_from_data/sigmoid.png){ align=center, width="500" }
/// caption
This simple plot shows how the sigmoid function takes inputs from `-10` to `10` and transforms them into a probability curve. The gray dashed line represents our decision boundary at 0.5.
///

- **Why this shape?** The sigmoid function gives us a smooth transition from 0 to 1, which is ideal for interpreting the output as probability. It's symmetric around `x = 0`, where `sigmoid(0) = 0.5`, making this the natural choice for our classification threshold.

- **Why is this useful?** When we're dealing with images of digits, this function allows us to convert the raw pixel data into something more interpretable—a probability that the image belongs to a particular class.

*By understanding this function, we gain insight into how logistic regression makes its classifications, turning raw data into decisions in a way that's both mathematically sound and intuitively understandable.*

### How Logistic Regression Works:

Logistic regression initially calculates a linear combination of the input features:

$$
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$

Where:

- $w_1, w_2, ..., w_n$ are the weights for each feature.
  
- $x_1, x_2, ..., x_n$ are the feature values.
  
- $b$ is the bias term.

Then, we apply the sigmoid function to this linear combination:

$$
\hat{y} = \sigma(z)
$$

Where:

- $\sigma(z)$ is our sigmoid function.

- $\hat{y}$ is the predicted probability that the input belongs to class 1.

Based on this probability, we can classify data points by setting a threshold (often 0.5):

- If $\hat{y} < 0.5$, predict class 0.

- If $\hat{y} \geq 0.5$, predict class 1.

#### Sigmoid in action

<iframe width="1707" height="765" src="https://www.youtube.com/embed/wLZnPYgbxdw" title="Sigmoid function in action" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

**Interpreting the Sigmoid Output:**

- As `x` goes to negative infinity, the output of the sigmoid function approaches 0.
- As `x` goes to positive infinity, the output approaches 1.
- A common threshold for classification is 0.5. If the output is less than 0.5, we might classify it as class 0, and if it's greater than 0.5, as class 1.

### Simulating Logistic Regression

Let's delve deeper into how logistic regression processes data to make predictions. We'll simulate this process using Python to illustrate the transformation from input to output.

**Setting Up Our Simulation:**
First, we define our parameters:

- **Number of Samples:** 100
- **Number of Features:** 3

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

num_samples, num_features = 100, 3
```

**Generating Weights, Bias, and Input Data:**

```python
# Initialize weights and bias randomly
weights = np.random.randn(num_features)
bias = np.random.randn()

# Create input data X
X = np.random.randn(num_samples, num_features)
```

**Computing the Linear Combination:**

```python
# Compute Z, which is our linear combination of weights and features plus bias
Z = X @ weights + bias  # Using @ for matrix multiplication
print(f"Shape of Z: {Z.shape}")  # This should be a 1-dimensional array of length 100
```

The `Z` array represents the linear combination of our input features with the weights, plus the bias. Here, `@` performs the dot product between `X` (100 samples by 3 features) and `weights` (3 features), resulting in a vector of 100 values, one for each sample.

**Applying the Sigmoid Function:**
Now we'll apply the sigmoid function to transform `Z` into probabilities:

```python
# Apply sigmoid to Z to get probabilities (y_hat)
y_hat = sigmoid(Z)
```

**Plotting the Results:**
We'll now plot the sigmoid curve along with our simulated predictions:

```python
# Generate data for the sigmoid plot
z_sigmoid = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z_sigmoid)

plt.figure(figsize=(10, 6))
# Plot the sigmoid function
plt.plot(z_sigmoid, sigmoid_values, label=r"$\sigma(z) = \frac{1}{1 + e^{-x}}$", color='blue')

# Plot our predicted values
plt.scatter(Z, y_hat, color="red", label="Predicted Values", alpha=0.8)
plt.axhline(y=0.5, linestyle="--", color="grey")  # Threshold line

plt.title("Simulated Predictions for Logistic Regression")
plt.xlabel("z")
plt.ylabel(r"$\hat{y}$")

plt.legend()
plt.tight_layout()

plt.show()
```

![Image title](../assets/dive_into_learning_from_data/log_reg_simmulation.png){ align=center, width="500" }
/// caption
This simulation visualizes how logistic regression uses the sigmoid function to convert a linear combination of features into class probabilities. Keep in mind, this is a simulation with random weights and bias, not an optimized model, but it helps to illustrate the concept.
///

**Explaining the Plot:**

- The blue line is the theoretical sigmoid function, showing how inputs are transformed into probabilities.
- The red dots represent our simulated model's predictions, where each dot corresponds to a sample's `Z` value mapped to its predicted probability `y_hat`.
- The horizontal grey dashed line at `y=0.5` is our decision boundary; points above this line would be classified as class 1, and below as class 0.
