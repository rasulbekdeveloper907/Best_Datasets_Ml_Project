import joblib
import pandas as pd
import gradio as gr

# ======================
# Model yuklash
# ======================
MODEL_PATH = r"C:\Users\Rasulbek907\Desktop\Best_Datasets_Ml_Project\Models\Simple_Models\RandomForestClassifier.joblib"
model = joblib.load(MODEL_PATH)


# ======================
# Predict function
# ======================
def predict(
    id,
    ref,
    subtitle,
    creatorname,
    creatorurl,
    totalbytes,
    url,
    lastupdated,
    downloadcount,
    ownername,
    ownerref,
    title,
    viewcount,
    tags
):
    df = pd.DataFrame([{
        "id": int(id),
        "ref": ref,
        "subtitle": subtitle,
        "creatorname": creatorname,
        "creatorurl": creatorurl,
        "totalbytes": int(totalbytes),
        "url": url,
        "lastupdated": lastupdated,
        "downloadcount": int(downloadcount),
        "ownername": ownername,
        "ownerref": ownerref,
        "title": title,
        "viewcount": int(viewcount),
        "tags": tags
    }])

    # Prediction
    predicted_cluster = int(model.predict(df)[0])

    # Probability (multiclass)
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(df)[0]
        confidence = float(proba_all[predicted_cluster])
    else:
        confidence = None

    return {
        "predicted_cluster": predicted_cluster,
        "confidence": round(confidence, 4) if confidence is not None else None
    }


# ======================
# Gradio UI
# ======================
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="ID", precision=0),
        gr.Textbox(label="Ref"),
        gr.Textbox(label="Subtitle"),
        gr.Textbox(label="Creator Name"),
        gr.Textbox(label="Creator URL"),
        gr.Number(label="Total Bytes", precision=0),
        gr.Textbox(label="URL"),
        gr.Textbox(label="Last Updated"),
        gr.Number(label="Download Count", precision=0),
        gr.Textbox(label="Owner Name"),
        gr.Textbox(label="Owner Ref"),
        gr.Textbox(label="Title"),
        gr.Number(label="View Count", precision=0),
        gr.Textbox(label="Tags")
    ],
    outputs=gr.JSON(label="Prediction Result"),
    title="Best Model Selection – Cluster Prediction",
    description="RandomForestClassifier | Predict cluster (0–4) from dataset metadata"
)

if __name__ == "__main__":
    demo.launch()
