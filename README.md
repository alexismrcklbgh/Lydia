# Lydia Transaction Classifier

## 1 Approach and Design Choices

The objective of this project is to automatically classify financial transactions into two levels of categories:
- **Major category**: broad transaction category.
- **Minor category**: more specific transaction category.

### Data preprocessing

The preprocessing pipeline is very simple:
- Split the dataset into training/test (80%/20%)
- Drop useless columns (all the same value/ all unique value (e.g transaction_id))
- Separate categorical and numerical features
- Check for missing features and remove lines in 
- Encode categorical features using One-Hot-Encoder 
- Concatenate back categorical and numerical fatures

### Machine Learning models

I choosed to train two separates Random Forest models, one to predict the minor_category, the other to predict the major_category.
Parameters of the Random Forest:

RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=10,
        min_samples_split=2,
        n_jobs=-1,
        class_weight="balanced"
    )

## 2. Results Analysis

The models were evaluated using classification metrics, including accuracy, precision, recall and F1-score.
We can clearly see that the dataset is unbalanced, more specific trainings with weights for each class could improve the overall performances
Other kind of model could also be tested as XGBoost or LightGBM. These are efficient model for classification problem

        Major_categories                precision    recall  f1-score   support

        entertainment                    0.88      0.94      0.91       393
        finance                          0.52      1.00      0.69        61
        groceries                        0.84      0.74      0.79       464
        refund                           1.00      1.00      1.00        28
        restaurants_bars_cafes           0.66      0.76      0.71       276
        self_care                        0.37      0.50      0.42        48
        shopping                         0.88      0.61      0.72        46
        transportation                   0.98      0.54      0.70       260
        travel                           0.32      0.78      0.45        32

        accuracy                            0.76      1608
        macro avg       0.72      0.76      0.71      1608
        weighted avg    0.81      0.76      0.77      1608

        Minor_categories                precision    recall  f1-score   support

        atm                             0.50      1.00      0.67        61
        betting                         0.73      0.93      0.82        95
        hairdresser                     1.00      0.95      0.98        21
        multimedia                      0.84      0.98      0.90       298
        others_shopping                 0.87      0.57      0.68        46
        pharmacies                      0.11      0.48      0.17        27
        refund                          1.00      1.00      1.00        28
        restaurants                     0.63      0.78      0.69       276
        supermarkets                    0.95      0.60      0.73       464
        tolls_gas_stations              1.00      0.63      0.77        91
        transport_means                 0.95      0.56      0.70       169
        travel_means                    0.86      0.75      0.80        32

        accuracy                            0.74      1608
        macro avg       0.79      0.77      0.74      1608
        weighted avg    0.83      0.74      0.76      1608


## 3. How to Run Training

### Requirements
Python 3.10 is recommended.

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Train the models

Run the training script:

```bash
python lydia.py
```

The training process performs preprocessing, trains both Random Forest classifiers, evaluates their performance and saves the following artifacts:

- `rfMajor.joblib`
- `rfMinor.joblib`
- `encoder.joblib`

Make sure the training dataset is available at the location expected by `lydia.py`.

## 4. Prediction Service

The prediction service is implemented using FastAPI.

### Project structure

```text
Lydia/
├── api.py
├── lydia.py
├── rfMajor.joblib
├── rfMinor.joblib
├── encoder.joblib
└── requirements.txt
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API

From the project directory, run:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The API documentation is available at:

`http://127.0.0.1:8000/docs`

### Prediction endpoint

**Method:** `POST`

**Endpoint:** `/predict`

**Content-Type:** `application/json`

The endpoint expects a single transaction as a JSON object.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "member_id": 49428667,
    "amount_value": -20.78,
    "fees": 0,
    "created_at": "2026-06-18T11:45:21Z",
    "merchant_id": "M6533702170",
    "merchant_name": "MAC DONALD S",
    "title": "MAC DONALD S",
    "card_name": "Carte sans contact basique",
    "digitalization_type": "apple_pay",
    "merchant_pan_entry_method": "CHIP_CONTACTLESS",
    "status": "completed"
  }'
```

### Expected input format

| Field | Type |
|---|---|
| member_id | Integer or null |
| amount_value | Float |
| fees | Integer or null |
| created_at | ISO 8601 datetime |
| merchant_id | String or null |
| merchant_name | String or null |
| title | String or null |
| card_name | String or null |
| digitalization_type | String or null |
| merchant_pan_entry_method | String or null |
| status | String or null |

### Example response

```json
{
  "major_category": "restaurants_bars_cafes",
  "minor_category": "restaurants"
}
```

For refund transactions, `major_category` is returned as `null`.

The API returns the predicted major and minor categories in JSON format.
