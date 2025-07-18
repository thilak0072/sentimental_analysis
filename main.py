from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import pandas as pd
import base64
import matplotlib.pyplot as plt
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

app = FastAPI()

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

class TextInput(BaseModel):
    text: list[str]

MAX_LENGTH = 512

@app.get("/")
async def index():
    return "API started successfully. Service is running."

@app.post("/predict_text")
async def predict_text(text_input: TextInput):
    try:
        predicted_sentiment = await single_prediction(model, tokenizer, text_input.text)
        return JSONResponse(content={
            "review": text_input.text,
            "prediction": predicted_sentiment
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
        if "Sentence" not in data.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'Sentence' column")
        predictions, graph = await bulk_prediction(model, tokenizer, data)

        graph.seek(0)
        graph_base64 = base64.b64encode(graph.getvalue()).decode("ascii")

        return StreamingResponse(
            predictions,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=Predictions.csv",
                "X-Graph-Exists": "true",
                "X-Graph-Data": graph_base64
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def single_prediction(model, tokenizer, text_input):
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    scores = softmax(logits.detach().numpy(), axis=1)
    predicted_classes = ["positive" if x[2] == max(x) else "negative" for x in scores]
    return predicted_classes

async def bulk_prediction(model, tokenizer, data):
    corpus = data['Sentence'].tolist()
    inputs = tokenizer(corpus, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.numpy()
    scores = softmax(logits.detach().numpy(), axis=1)
    predicted_classes = ["positive" if x[2] == max(x) else "negative" for x in scores]
    data["Predicted sentiment"] = predicted_classes

    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = await get_distribution_graph(data)
    return predictions_csv, graph

async def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution"
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()
    return graph

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)