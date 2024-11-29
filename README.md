# Sentiment Analysis with PyTorch

A deep learning-based sentiment analysis implementation using PyTorch. The model uses bigram-based text processing and a neural network architecture to classify text sentiment.

## Features

- Bigram-based text representation
- Custom PyTorch Dataset implementation
- Neural network with embedding layer
- Support for both training and inference
- Automatic vocabulary building
- GPU support when available
- Text preprocessing and cleaning

## Requirements

```
torch>=1.0.0
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install torch
```

## Usage

The script supports two main modes: training and testing.

### Command Line Arguments

- `--text_path`: Path to the input text file
- `--label_path`: Path to the label file (required for training)
- `--train`: Flag to run in training mode
- `--test`: Flag to run in testing mode
- `--model_path`: Path where model will be saved (training) or loaded from (testing)
- `--output_path`: Path where predictions will be saved during testing (default: 'out.txt')

### Training Mode

To train a new model:

```bash
python sentiment_model.py --train \
    --text_path path/to/training/texts.txt \
    --label_path path/to/training/labels.txt \
    --model_path path/to/save/model.pt
```

### Testing Mode

To use a trained model for prediction:

```bash
python sentiment_model.py --test \
    --text_path path/to/test/texts.txt \
    --model_path path/to/saved/model.pt \
    --output_path path/to/predictions.txt
```

## Model Architecture

### Text Processing

- Converts text to lowercase
- Removes HTML tags
- Removes punctuation and numbers
- Uses bigram-based tokenization

### Neural Network

- Embedding layer (dimension: 20)
- Linear layer with ReLU activation
- Dropout layer (p=0.5)
- Output layer with sigmoid activation
- Binary Cross-Entropy loss function

### Training Parameters

- Batch size: 32
- Learning rate: 5e-3
- Number of epochs: 10
- Optimizer: Adam

## Dataset Format

### Input Text File

- One sentence per line
- Plain text format
- Will be automatically preprocessed

### Label File

- One label per line
- Binary labels (0 or 1)
- Must align with input text file

## Model Output

The model generates binary predictions (0 or 1) based on a threshold of 0.5. These predictions are saved to the specified output file, one prediction per line.

## Implementation Details

### Key Components

1. `SentDataset` Class:

   - Custom PyTorch Dataset implementation
   - Handles text preprocessing
   - Builds vocabulary from training data
   - Manages bigram creation and indexing

2. `Model` Class:

   - Neural network architecture
   - Implements masked averaging of embeddings
   - Handles padding in batched inputs

3. Training Function:

   - Implements training loop
   - Supports GPU acceleration
   - Saves model and vocabulary
   - Reports training progress

4. Testing Function:
   - Batch prediction
   - Threshold-based classification
   - Saves predictions to file

## Error Handling

The implementation includes checks for:

- Required command line arguments
- GPU availability
- File existence and accessibility
- Model path during testing
- Label file presence during training

## Notes

- The model uses bigrams instead of individual words to capture some local context
- Zero padding is used for batch processing
- The vocabulary is built during training and saved with the model
- The model automatically uses GPU if available

## Performance Considerations

- Memory usage scales with vocabulary size
- Processing time depends on input text length and batch size
- GPU acceleration can significantly improve training speed

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
