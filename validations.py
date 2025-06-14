def validate_structured_info(structured_info, reference_info, threshold):
    """
    Dummy validation function. Compares fields in structured_info and reference_info.
    Returns a dict with 'field_results' as expected by the app.
    """
    field_results = {}
    # List of fields to check
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

        # Simple match logic
        if extracted_value is None or reference_value is None:
            status = "null"
            score = 0
        elif str(extracted_value).strip().lower() == str(reference_value).strip().lower():
            status = "match"
            score = 100
        else:
            status = "hallucination"
            score = 0

        field_results[field] = {
            "status": status,
            "score": score,
            "extracted_value": extracted_value,
            "reference_value": reference_value,
        }
    return {"field_results": field_results}