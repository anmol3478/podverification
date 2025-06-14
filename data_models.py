from typing import Optional, Dict, Any
from pydantic import BaseModel

class TextLabel(BaseModel):
    text: Optional[str]
    box_2d: Optional[list]

class StructuredImageProperty(BaseModel):
    # This is a flexible model for fields like courier_partner, awb_number, etc.
    text_quality_score: Optional[Any] = None
    courier_partner: Optional[Dict[str, Any]] = None
    awb_number: Optional[Dict[str, Any]] = None
    recipient_name: Optional[Dict[str, Any]] = None
    recipient_address: Optional[Dict[str, Any]] = None
    recipient_signature: Optional[Dict[str, Any]] = None
    recipient_stamp: Optional[Dict[str, Any]] = None
    delivery_date: Optional[Dict[str, Any]] = None
    handwritten_notes: Optional[Dict[str, Any]] = None

class ImageMaster(BaseModel):
    image_url: Optional[str] = None
    image: Optional[Any] = None
    structured_info: Optional[StructuredImageProperty] = None
    reference_info: Optional[Dict[str, Any]] = None

    @classmethod
    def model_validate_json(cls, json_string: str):
        import json
        data = json.loads(json_string)
        # Parse structured_info as StructuredImageProperty if present
        if "structured_info" in data and isinstance(data["structured_info"], dict):
            data["structured_info"] = StructuredImageProperty(**data["structured_info"])
        return cls(**data)