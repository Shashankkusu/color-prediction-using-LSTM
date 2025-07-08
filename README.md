# Color Name to RGB Prediction using LSTM

This project demonstrates how to use an LSTM (Long Short-Term Memory) neural network to predict the RGB values of a color given its name. The notebook preprocesses a dataset of color names and their RGB values, tokenizes the names at the character level, and trains an LSTM-based model to learn the mapping from name to color.

---

## 🖼️ LSTM Architecture

Below is an image illustrating the general architecture of an LSTM network used for this task:

![LSTM Architecture](https://raw.githubusercontent.com/rasbt/deeplearning-models/master/img/lstm_architecture.png)

*Image source: Sebastian Raschka (https://github.com/rasbt/deeplearning-models)*

---

## Project Workflow

1. **Data Preparation**
    - Color names and their RGB values are loaded from a CSV file.
    - Names are tokenized at the character level and padded to a maximum length.
    - RGB values are normalized to the [0, 1] range.

2. **Model Architecture**
    - Input: One-hot encoded color name (shape: batch, maxlen, num_characters)
    - LSTM Layer 1: 256 units, return sequences
    - LSTM Layer 2: 128 units
    - Dense Layer: 128 units, ReLU activation
    - Output Dense Layer: 3 units (for R, G, B channels), sigmoid activation

3. **Training**
    - The model is trained to minimize mean squared error (MSE) between predicted and true normalized RGB values.
    - Accuracy is measured as the percentage of exact matches (rounded to nearest integer).

---

## How Does LSTM Predict the RGB Value from a Color Name?

### 1. **Character Encoding**
The color name is first converted into a sequence of integers based on a character-level tokenizer. For example, `"blue"` might become `[7, 12, 5, 9]`, where each number corresponds to a character.

### 2. **One-Hot Encoding and Padding**
Each integer is one-hot encoded into a binary vector. The sequence is padded to a fixed length (e.g., 25 characters) to ensure uniform input shape.

### 3. **Sequence Processing by LSTM**
The LSTM layers process the sequence of one-hot vectors. LSTMs are well-suited for handling sequences because they can learn patterns over varying input lengths and retain context with their internal memory (cell state).

- **First LSTM Layer**: Processes the sequence, outputting a representation for each character position.
- **Second LSTM Layer**: Further condenses this information into a fixed-size vector that summarizes the entire color name.

### 4. **Dense Layers for Prediction**
The output of the LSTM is passed through dense (fully connected) layers, which learn to map the extracted features to three values: the normalized red, green, and blue channel intensities.

### 5. **Output**
The final sigmoid output produces three numbers between 0 and 1, which are then scaled to [0, 255] to get the RGB values.

### 6. **Why LSTM Works Here**
LSTM can capture:
- Prefixes/suffixes (e.g., "light", "dark", "ish")
- Contextual character patterns (e.g., "-green", "-blue")
- Repeated structures and character dependencies

This enables the model to generalize to new, unseen color names based on learned language patterns.

---

## Example Prediction

Given the name `"light blue"`, the prediction process is:

1. Tokenize and pad: `["l", "i", "g", "h", "t", " ", "b", "l", "u", "e", ...]` → padded sequence
2. One-hot encode
3. Pass through LSTM layers
4. Dense layers output `[0.7, 0.8, 0.9]` (example values)
5. Scale to `[178, 204, 230]` (RGB)
6. Display the color as an image patch

---

## Usage

To use the model for your own color name:

```python
predict("forest green")
```

---

## Requirements

- Python 3.6+
- TensorFlow >= 2.0
- pandas, numpy, scipy, matplotlib

---

## References

- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Color Names Dataset](https://github.com/meodai/color-names)
- [Sebastian Raschka LSTM Diagram](https://github.com/rasbt/deeplearning-models)

```
(https://raw.githubusercontent.com/rasbt/deeplearning-models/master/img/lstm_architecture.png)
```

---
