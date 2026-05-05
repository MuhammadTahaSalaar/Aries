"""
ARIES — BART (Seq2Seq) ONNX export.

Exports the fine-tuned BART model to ONNX for inference deployment.
BART is an encoder-decoder model so we export the encoder and decoder
as a single model with the generate-compatible wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_bart_to_onnx(
    model_dir: str | Path,
    output_path: str | Path,
    opset: int = 17,
    optimise: bool = True,
) -> Path:
    """
    Export a fine-tuned BART model to ONNX.

    For encoder-decoder models we rely on the Optimum OnnxForSeq2SeqLM
    exporter which handles the dual-graph (encoder + decoder).

    Parameters
    ----------
    model_dir   : Directory containing the saved BART + tokeniser files.
    output_path : Destination directory for the ONNX files.
    opset       : ONNX opset version (default 17).
    optimise    : Run ONNX Runtime transformer optimisation.

    Returns
    -------
    Path to the output directory containing ONNX artifacts.
    """
    import torch
    import onnx
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading BART from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    model.eval()

    # ── Export encoder ────────────────────────────────────────────────
    encoder = model.get_encoder()
    enc_out_path = output_path / "encoder.onnx"

    dummy_input_ids = torch.zeros(1, 64, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 64, dtype=torch.long)

    logger.info("Exporting encoder → %s (opset %d)", enc_out_path, opset)
    torch.onnx.export(
        encoder,
        (dummy_input_ids, dummy_attention_mask),
        str(enc_out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(enc_out_path)))
    logger.info("Encoder ONNX valid ✓  size=%d bytes", enc_out_path.stat().st_size)

    # ── Export decoder ────────────────────────────────────────────────
    # Use the full model's forward for decoder (with encoder_outputs)
    dec_out_path = output_path / "decoder.onnx"
    dummy_decoder_ids = torch.zeros(1, 16, dtype=torch.long)
    dummy_encoder_hidden = torch.randn(1, 64, model.config.d_model)

    # Create a decoder wrapper for clean ONNX export
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.get_decoder()
            self.lm_head = model.lm_head
            self.final_logits_bias = model.final_logits_bias

        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
            dec_out = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            logits = self.lm_head(dec_out.last_hidden_state) + self.final_logits_bias
            return logits

    decoder_wrapper = DecoderWrapper(model)
    decoder_wrapper.eval()

    logger.info("Exporting decoder → %s (opset %d)", dec_out_path, opset)
    torch.onnx.export(
        decoder_wrapper,
        (dummy_decoder_ids, dummy_encoder_hidden, dummy_attention_mask),
        str(dec_out_path),
        input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "decoder_input_ids": {0: "batch", 1: "dec_seq_len"},
            "encoder_hidden_states": {0: "batch", 1: "enc_seq_len"},
            "encoder_attention_mask": {0: "batch", 1: "enc_seq_len"},
            "logits": {0: "batch", 1: "dec_seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(dec_out_path)))
    logger.info("Decoder ONNX valid ✓  size=%d bytes", dec_out_path.stat().st_size)

    # ── Optimise with ORT (optional) ──────────────────────────────────
    if optimise:
        try:
            from onnxruntime.transformers.optimizer import optimize_model

            # ORT optimizer supports "bert" for encoder-style sub-graphs;
            # "bart" is not a valid type. Only optimise the encoder.
            opt = optimize_model(
                str(enc_out_path),
                model_type="bert",
                opt_level=1,
                use_gpu=False,
            )
            opt_path = enc_out_path.with_suffix(".opt.onnx")
            opt.save_model_to_file(str(opt_path))
            logger.info("Optimised encoder → %s", opt_path.name)

        except ImportError:
            logger.warning(
                "onnxruntime.transformers not available — skipping optimisation"
            )
        except Exception:
            logger.warning("ONNX optimisation failed — using unoptimised model",
                           exc_info=True)

    # ── Save tokeniser alongside ──────────────────────────────────────
    tokenizer.save_pretrained(str(output_path))

    logger.info("BART ONNX export complete → %s", output_path)
    return output_path


def validate_bart_onnx(
    onnx_dir: str | Path,
    sample_text: str = "The malware was designed to exploit a vulnerability in the system.",
) -> dict[str, bool]:
    """Quick sanity check: run encoder → decoder → compare shapes."""
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_dir = Path(onnx_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))

    inputs = tokenizer(sample_text, return_tensors="np", padding=True)

    # Run encoder
    enc_path = onnx_dir / "encoder.opt.onnx"
    if not enc_path.exists():
        enc_path = onnx_dir / "encoder.onnx"

    enc_sess = ort.InferenceSession(str(enc_path))
    enc_out = enc_sess.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )

    # Run decoder with BOS token
    dec_path = onnx_dir / "decoder.opt.onnx"
    if not dec_path.exists():
        dec_path = onnx_dir / "decoder.onnx"

    dec_sess = ort.InferenceSession(str(dec_path))
    decoder_input_ids = np.array([[tokenizer.bos_token_id or 0]], dtype=np.int64)

    dec_out = dec_sess.run(
        None,
        {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": enc_out[0],
            "encoder_attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )

    logits = dec_out[0]
    logger.info(
        "BART ONNX validation — encoder_out=%s, decoder_logits=%s",
        enc_out[0].shape, logits.shape,
    )

    return {
        "encoder_valid": enc_out[0].ndim == 3,
        "decoder_valid": logits.ndim == 3 and logits.shape[-1] > 1000,
        "encoder_shape": list(enc_out[0].shape),
        "decoder_shape": list(logits.shape),
    }
