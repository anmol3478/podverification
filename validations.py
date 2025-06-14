import difflib

def validate_structured_info(structured_info, reference_info, threshold):
    """
    Validation function. Compares fields in structured_info and reference_info.
    Assigns a score based on similarity between extracted and reference values.
    Uses the threshold to determine match/hallucination/null status.
    """
    field_results = {}
    fields = [
        "text_quality_score", "courier_partner", "awb_number", "recipient_name",
        "recipient_address", "recipient_signature", "recipient_stamp", "delivery_date", "handwritten_notes"
    ]
    for field in fields:
        extracted = getattr(structured_info, field, None)
        ref = reference_info.get(field) if reference_info else None

        # Extract text if it's a dict with 'text'
        extracted_value = extracted.get("text") if isinstance(extracted, dict) and "text" in extracted else extracted
        reference_value = ref

        if extracted_value is None or reference_value is None:
            status = "null"
            score = 0
        else:
            # Compute similarity ratio (0 to 1)
            ratio = difflib.SequenceMatcher(None, str(extracted_value).strip().lower(), str(reference_value).strip().lower()).ratio()
            score = int(ratio * 100)
            if score >= threshold:
                status = "match"
            else:
                status = "hallucination"

        field_results[field] = {
            "status": status,
            "score": score,
            "extracted_value": extracted_value,
            "reference_value": reference_value,
        }
    return {"field_results": field_results}
