# Toxic Comment Classification API

A Flask API for predicting toxic comments using trained machine learning models.

## Endpoints

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Check if the API is running and which models are loaded

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": ["tfidf_logistic", "tfidf_svm", "tfidf_random_forest", "bert"]
}
```

### Single Prediction
- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Predict toxicity for a single comment

**Request Body**:
```json
{
  "text": "Your comment text here",
  "model": "tfidf_logistic"  // optional, defaults to tfidf_logistic
}
```

**Available Models**:
- `tfidf_logistic`
- `tfidf_svm`
- `tfidf_random_forest`
- `bert`

**Response**:
```json
{
  "text": "Your comment text here",
  "model": "tfidf_logistic",
  "probabilities": {
    "toxic": 0.123,
    "severe_toxic": 0.045,
    "obscene": 0.067,
    "threat": 0.012,
    "insult": 0.089,
    "identity_hate": 0.034
  },
  "predictions": {
    "toxic": 0,
    "severe_toxic": 0,
    "obscene": 0,
    "threat": 0,
    "insult": 0,
    "identity_hate": 0
  }
}
```

### Batch Prediction
- **URL**: `/batch_predict`
- **Method**: `POST`
- **Description**: Predict toxicity for multiple comments

**Request Body**:
```json
{
  "texts": ["Comment 1", "Comment 2", "Comment 3"],
  "model": "bert"  // optional, defaults to tfidf_logistic
}
```

**Response**: Array of prediction results (same format as single prediction)

## Running the API

1. Activate the conda environment:
```bash
conda activate toxbench
```

2. Start the server:
```bash
python scripts/app.py
```

3. The API will be available at `http://localhost:5001`

## Usage Examples

### Python
```python
import requests

# Single prediction
response = requests.post('http://localhost:5001/predict',
                        json={'text': 'You are awesome!', 'model': 'bert'})
print(response.json())

# Batch prediction
response = requests.post('http://localhost:5001/batch_predict',
                        json={'texts': ['Nice work!', 'This sucks!'], 'model': 'tfidf_svm'})
print(response.json())
```

### Command Line
```bash
# Health check
curl http://localhost:5001/health

# Single prediction
curl -X POST http://localhost:5001/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Great job!", "model": "tfidf_logistic"}'

# Batch prediction
curl -X POST http://localhost:5001/batch_predict \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Good work", "Bad work"], "model": "bert"}'
```