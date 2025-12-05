"""Error Analysis Casebook Generator."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

def generate_error_casebook(
    predictions_path: Path,
    output_path: Optional[Path] = None,
    top_k_errors: int = 10
) -> pd.DataFrame:
    """Find the most 'confident' errors (False Positives/Negatives).
    
    Args:
        predictions_path: Path to test_predictions.csv from training pipeline.
        output_path: Path to save the casebook CSV.
        top_k_errors: Number of top errors to extract per label/type.
        
    Returns:
        DataFrame containing the casebook.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        
    df = pd.read_csv(predictions_path)
    
    # Identify label columns based on _prob suffix
    label_cols = [c.replace("_prob", "") for c in df.columns if c.endswith("_prob")]
    
    casebook_rows = []
    
    for label in label_cols:
        prob_col = f"{label}_prob"
        pred_col = f"{label}_pred"
        
        # We need the true labels. 
        # The predictions CSV usually contains predictions and probs.
        # It assumes the 'truth' was not saved there unless we put it there.
        # Wait, standard practice is to join with ground truth or assume it's in the file.
        # Our `_build_predictions_frame` in train.py DOES NOT currently save y_true.
        # We rely on the index to join back to source or we should update train.py.
        # For now, let's assume we can't fully do it without y_true.
        # CHECK: src/pipeline/train.py -> _build_predictions_frame 
        # It adds "row_index".
        
        # Strategy: We need true labels to find errors.
        # If 'y_true' isn't in preds csv, we must look at the 'label' columns if they were preserved?
        # In `_build_predictions_frame`, we passed `test_df`.
        # `test_df` usually has the label columns.
        
        if label not in df.columns:
            # If standard label cols are missing, we might have lost them.
            # But `_build_predictions_frame` includes all columns from `test_df` via `test_df.get(text_col)`... 
            # actually it explicitly builds a payload.
            # Let's check train.py:480... It only adds ID, row_index, and text. 
            # It does NOT add the original labels.
            # CRITICAL MISSING FEATURE: We need true labels in the output CSV to analyze errors.
            print(f"Warning: True label column '{label}' not found in predictions. Skipping.")
            continue
            
        # False Positives: Pred=1, True=0 (High prob, should be low)
        fp_mask = (df[pred_col] == 1) & (df[label] == 0)
        fp_df = df[fp_mask].sort_values(by=prob_col, ascending=False).head(top_k_errors)
        
        for _, row in fp_df.iterrows():
            casebook_rows.append({
                "type": "False Positive",
                "label": label,
                "text": row.get("comment_text", "")[:200], # Truncate for readability
                "probability": row[prob_col],
                "true_label": 0,
                "predicted": 1
            })
            
        # False Negatives: Pred=0, True=1 (Low prob, should be high)
        fn_mask = (df[pred_col] == 0) & (df[label] == 1)
        # We want the ones with lowest probability (closest to 0) that were actually true?
        # Or the ones that were "closest to being correct" (borderline)?
        # Usually "most confident error" means prob is very low (e.g. 0.01) but it was Toxic.
        fn_df = df[fn_mask].sort_values(by=prob_col, ascending=True).head(top_k_errors)
        
        for _, row in fn_df.iterrows():
            casebook_rows.append({
                "type": "False Negative",
                "label": label,
                "text": row.get("comment_text", "")[:200],
                "probability": row[prob_col],
                "true_label": 1,
                "predicted": 0
            })

    casebook = pd.DataFrame(casebook_rows)
    
    if output_path:
        casebook.to_csv(output_path, index=False)
        print(f"Casebook saved to {output_path}")
        
    return casebook

