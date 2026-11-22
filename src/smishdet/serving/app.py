"""Gradio demo.

  python -m smishdet.serving.app

Binds to 127.0.0.1 only. Messages typed here are not stored or logged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smishdet.data.synthetic import render
from smishdet.domain.schema import Label
from smishdet.models.base import Classifier
from smishdet.serving.predictor import Analysis, analyze, load_classifiers

DISCLAIMER = (
    "**Research demo, not a security product.** Never rely on it to decide whether a message "
    "is safe. The model was trained mostly on English SMS plus machine-translated and "
    "template-generated Sinhala/Tamil text, and has not been tested on real Sinhala/Tamil "
    "messages yet. Verify suspicious messages with the official organisation, and never share "
    "OTPs, NIC or card details. Messages you enter are not stored."
)

VERDICT = {
    Label.SMISHING: "Likely a scam / phishing message (smishing)",
    Label.SPAM: "Likely unwanted promotional message (spam)",
    Label.HAM: "Looks like a normal message (ham)",
}

# Examples are rendered from the synthetic templates so they always match the generator.
EXAMPLES = [
    [render("traffic1", "en", 1)],
    [render("bank1", "si", 2)],
    [render("prize2", "ta", 3)],
    [render("otp2", "si-en", 1)],
    [render("customs1", "ta-en", 4)],
    [render("ham1", "en", 5)],
    [render("ham4", "si", 0)],
]


def explain(a: Analysis) -> str:
    """Markdown summary shown under the label bars."""
    lines = [f"**{VERDICT[a.label]}**, confidence {a.confidence:.0%}."]
    if a.uncertain:
        lines.append("The model is **not confident**: treat this result as a guess.")
    lines.append(f"Detected script: `{a.script}`.")
    if a.processed.count("<URL>") or a.processed.count("<PHONE>"):
        lines.append("Links and phone numbers are replaced by `<URL>` / `<PHONE>` before "
                     "scoring, exactly as in the training data.")
    return "\n\n".join(lines)


def build_app(models: dict[str, Classifier]):  # type: ignore[no-untyped-def]
    import gradio as gr

    def run(text: str, model_name: str) -> tuple[dict[str, float], str]:
        try:
            a = analyze(text, models[model_name])
        except ValueError:
            return {}, "Enter a message to analyse."
        return {lab.value: p for lab, p in a.probs.items()}, explain(a)

    names = list(models)
    with gr.Blocks(title="Sinhala/Tamil scam SMS detector") as demo:
        gr.Markdown("# Scam & phishing SMS detector\n"
                    "Sinhala, Tamil, English, Singlish and Tanglish.")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Message", lines=5,
                                  placeholder="Paste an SMS here (Sinhala, Tamil, English...)")
                model = gr.Dropdown(names, value=names[0], label="Model")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column():
                label = gr.Label(label="Probabilities", num_top_classes=3)
                info = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=[text], label="Examples (generated from templates)")
        btn.click(run, [text, model], [label, info])
        text.submit(run, [text, model], [label, info])
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tfidf", type=Path, default=Path("outputs/tfidf_multi_all/model.joblib"))
    p.add_argument("--xlmr", type=Path, default=Path("outputs/xlmr_multi_all/best"))
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    models = load_classifiers(args.tfidf, args.xlmr)
    print("loaded:", list(models))
    build_app(models).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
