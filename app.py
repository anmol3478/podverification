import streamlit as st
import pandas as pd
import json
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import logging
from typing import List, Dict, Any, Optional
from validations import validate_structured_info

# Assuming data_models.py is in the same directory or accessible via PYTHONPATH
try:
    from data_models import ImageMaster, TextLabel, StructuredImageProperty
except ImportError:
    st.error("Error: Could not import data models. Make sure src/data_models.py exists and is accessible.")
    st.stop()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def draw_predictions_on_image(image: Image.Image, structured_info: Optional[StructuredImageProperty], font_size: int = 15) -> Image.Image:
    """Draws bounding boxes and labels from StructuredImageProperty onto the image."""
    if not structured_info:
        return image

    draw = ImageDraw.Draw(image)
    try:
        # Load a default font or specify a path
        # Use a common font likely available on macOS/Linux/Windows
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size) # Case variation
            except IOError:
                 try:
                     font = ImageFont.truetype("DejaVuSans.ttf", font_size) # Common on Linux
                 except IOError:
                    font = ImageFont.load_default()
                    logging.warning("Common fonts (Arial, DejaVuSans) not found, using PIL default font. Labels might look basic.")

    except Exception as e: # Catch any other font loading errors
        font = ImageFont.load_default()
        logging.warning(f"Error loading font: {e}. Using PIL default font.")


    structured_info_dict = structured_info.model_dump()
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    color_index = 0

    for field_name, field_value in structured_info_dict.items():
        #if field_name!='handwritten_notes':

            if isinstance(field_value, dict) and 'text' in field_value and 'box_2d' in field_value:
                text_label_data = field_value
                text = text_label_data.get('text', '')
                box_2d = text_label_data.get('box_2d')

                # 7. Validate Bounding Box: Check if box_2d exists, is a list, and has exactly 4 coordinates.
                if box_2d and isinstance(box_2d, list) and len(box_2d) == 4:
                    img_width, img_height = image.size
                    ## Normal Method
                    # abs_x1, abs_y1, abs_x2, abs_y2 = map(int, box_2d)

                    ## Gemini Bounding Box processing method
                    scale = 1000 
                    abs_y1 = int(box_2d[1] / scale * img_height)
                    abs_x1 = int(box_2d[0] / scale * img_width)
                    abs_y2 = int(box_2d[3] / scale * img_height)
                    abs_x2 = int(box_2d[2] / scale * img_width)

                    # Ensure coordinates are ordered correctly (x1 <= x2, y1 <= y2)
                    if abs_x1 > abs_x2:
                        abs_x1, abs_x2 = abs_x2, abs_x1
                    if abs_y1 > abs_y2:
                        abs_y1, abs_y2 = abs_y2, abs_y1

                    # Optional: Clamp coordinates to image boundaries after scaling and swapping (Safety measure)
                    abs_x1 = max(0, min(abs_x1, img_width))
                    abs_y1 = max(0, min(abs_y1, img_height))
                    abs_x2 = max(0, min(abs_x2, img_width))
                    abs_y2 = max(0, min(abs_y2, img_height))

                    # Skip if box is invalid after processing (e.g., zero width/height)
                    if abs_x1 >= abs_x2 or abs_y1 >= abs_y2:
                        logging.warning(f"Skipping invalid bbox for {field_name}: {[abs_x1, abs_y1, abs_x2, abs_y2]}")
                        continue

                    # Select Color
                    color = colors[color_index % len(colors)]
                    color_index += 1

                    # Draw Bounding Box
                    # Note: Using ((x1, y1), (x2, y2)) format for draw.rectangle
                    draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3) # Width=3 matches original

                    # Prepare Label Text: Create the label string (e.g., "field_name: detected_text"). Truncate long text.
                    label = f"{field_name}: {text[:15]}{'...' if len(text)>15 else ''}" # Show field name and truncated text
                    # Calculate text dimensions using getbbox
                    try:
                        # getbbox returns (left, top, right, bottom) relative to origin
                        text_bbox = font.getbbox(label)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        # Adjust for potential negative top offset in some fonts
                        text_y_offset = text_bbox[1] # Accounts for font rendering specifics
                    except AttributeError: # Fallback if font object doesn't have getbbox
                        text_width, text_height, text_y_offset = (0, 0, 0)
                        logging.warning("Font object missing getbbox, cannot determine text size accurately.")
                    except Exception as e: # Catch other potential errors
                        text_width, text_height, text_y_offset = (0, 0, 0)
                        logging.warning(f"Error getting text bbox: {e}")


                    # 13. Draw Text Background:
                    #     - Calculate the position for a filled rectangle (using the chosen color) to act as a background for the text.
                    #     - Position it slightly above the bounding box (using abs_y1).
                    # Adjust y position based on text_height and potential negative offset
                    text_bg_y = max(0, abs_y1 - text_height - text_y_offset - 2) # Position above box, ensure non-negative
                    # Ensure the background rectangle width calculation uses the calculated text_width
                    draw.rectangle([abs_x1, text_bg_y, abs_x1 + text_width + 4, text_bg_y + text_height + 2], fill=color)

                    # 14. Draw Text: Draw the label text (in black) on top of the background rectangle.
                    # Adjust text drawing position based on offset
                    draw.text((abs_x1 + 2, text_bg_y - text_y_offset + 1), label, fill="black", font=font)
                else:
                    # Log if a field is skipped due to bad box data.
                    logging.warning(f"Skipping field '{field_name}' due to missing or invalid box_2d: {box_2d}")

    return image

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Document Visualization Tool")

st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
json_column = st.sidebar.text_input("Column with ImageMaster JSON", "output")
# Optional: Add flexibility for image source if not in ImageMaster JSON
image_source_column = st.sidebar.text_input("Column with Image URL/Path (if not in JSON)", "image_url")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} rows from {uploaded_file.name}")

        if json_column not in df.columns:
            st.error(f"Error: JSON column '{json_column}' not found in the CSV. Available: {list(df.columns)}")
            st.stop()
        if image_source_column and image_source_column not in df.columns:
             st.warning(f"Image source column '{image_source_column}' not found. Will rely on 'image_url' within the JSON.")
             # Don't stop, just warn.

        # --- Sample Selection ---
        if 'selected_index' not in st.session_state:
            st.session_state.selected_index = 0
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state.selected_index = max(0, st.session_state.selected_index - 1)
        with col2:
            if st.button("Next"):
                st.session_state.selected_index = min(len(df) - 1, st.session_state.selected_index + 1)

        # Add a way to select which row to view
        st.sidebar.header("Select Image")
        st.session_state.selected_index = st.sidebar.selectbox("Choose a row index:", df.index, index=st.session_state.selected_index)


        # --- Validation Threshold Configuration ---
        st.sidebar.header("Benchmarking Mode")
        benchmarking_mode = st.sidebar.radio(
            "Select Benchmarking Mode:",
            ("Sample-wise", "Overall"),
            index=0,  # Default to Sample-wise
        )

        st.sidebar.header("Validation Settings")
        validation_threshold = st.sidebar.slider(
            "Fuzzy Match Threshold (%)", 0, 100, 50, 5
        )

        st.header(f"Displaying Row: {st.session_state.selected_index}")
        selected_row = df.iloc[st.session_state.selected_index]

        if benchmarking_mode == "Overall":
            # --- Overall Statistics Calculation ---
            overall_data = []
            for column in ["text_quality_score", "courier_partner", "awb_number", "recipient_name", "recipient_address", "recipient_signature", "recipient_stamp", "delivery_date"]:
                total_count = len(df)
                total_null = 0
                hallucination_count = 0
                total_match = 0

                # Calculate match and hallucination count
                for index, row in df.iterrows():
                    try:
                        json_string = row[json_column]
                        if pd.isna(json_string):
                            continue  # Skip if JSON is empty

                        image_master = ImageMaster.model_validate_json(json_string)

                        if image_master.structured_info and image_master.reference_info:
                            validation_results = validate_structured_info(
                                image_master.structured_info,
                                image_master.reference_info,
                                validation_threshold,
                            )
                            if column in validation_results["field_results"]:
                                field_result = validation_results["field_results"][column]
                                if field_result["status"] == "match":
                                    total_match += 1
                                elif field_result["status"] == "hallucination":
                                    hallucination_count += 1
                                else:
                                    total_null +=1
                            else:
                                total_null +=1
                    except Exception as e:
                        logging.error(f"Error processing row {index}: {e}")
                        continue

                match_percentage = (total_match / total_count) * 100 if total_count else 0
                null_percentage = (total_null / total_count) * 100 if total_count else 0
                hallucination_percentage = (hallucination_count / total_count) * 100 if total_count else 0

                overall_data.append({
                    "Field": column,
                    "Total Count": total_count,
                    "Total Match": total_match,
                    "Total Null": total_null,
                    "Hallucination Count": hallucination_count,
                    "Match Percentage": f"{match_percentage:.2f}%",
                    "Null Percentage": f"{null_percentage:.2f}%",
                    "Hallucination Percentage": f"{hallucination_percentage:.2f}%",
                })

            overall_df = pd.DataFrame(overall_data)

            st.subheader("Overall Statistics")
            st.dataframe(overall_df)

        else:  # Sample-wise
            # --- Parse ImageMaster JSON ---
            image_master = None
            json_string = ""
            try:
                json_string = selected_row[json_column]
                if pd.isna(json_string):
                    st.error(f"JSON data in column '{json_column}' for row {st.session_state.selected_index} is empty or NaN.")
                    st.stop()
                image_master = ImageMaster.model_validate_json(json_string)

                # --- Validate Structured Info ---
                if image_master.structured_info and image_master.reference_info:
                    validation_results = validate_structured_info(
                        image_master.structured_info,
                        image_master.reference_info,
                        validation_threshold,
                    )
                    field_results = validation_results["field_results"]

                    # Create a Pandas DataFrame for the validation results
                    validation_data = []
                    for field, result in field_results.items():
                        validation_data.append({
                            "Field": field,
                            "Status": result.get("status", "unknown"),
                            "Score": result.get("score", 0),
                            "Extracted Value": result.get("extracted_value", ""),
                            "Reference Value": result.get("reference_value", ""),
                        })
                    validation_df = pd.DataFrame(validation_data)

                    st.subheader("Validation Results")
                    st.dataframe(validation_df)

                st.subheader("Extracted Information")
                st.json(
                    image_master.structured_info.model_dump_json(
                        indent=2
                    )
                    if image_master.structured_info
                    else {},
                    expanded=False,
                )
            except Exception as e:
                st.error(f"Error parsing ImageMaster JSON in row {selected_index}: {e}")
                st.write("JSON Content (first 500 chars):")
                st.text(str(json_string)[:500])  # Show the problematic JSON (truncated)
                st.stop()

            # --- Get Image URL/Path ---
            img_src = None
            # Priority 1: From ImageMaster JSON
            if image_master and image_master.image_url:
                img_src = image_master.image_url
            # Priority 2: From the specified image source column
            elif image_source_column and image_source_column in df.columns and pd.notna(selected_row[image_source_column]):
                img_src = selected_row[image_source_column]
            else:
                st.error(f"Could not find image source. Checked ImageMaster JSON ('image_url') and column '{image_source_column}' for row {selected_index}.")
                st.stop()


            # --- Load and Display Image ---
            if img_src:
                st.subheader("Image with Predictions")
                image = None
                try:
                    # Handle potential local file paths vs URLs
                    if str(img_src).startswith(('http://', 'https://')):
                        headers = {'User-Agent': 'Mozilla/5.0'} # Add a user-agent header
                        response = requests.get(img_src, stream=True, headers=headers, timeout=10) # Add timeout
                        response.raise_for_status() # Raise an exception for bad status codes
                        # Check content type if possible
                        content_type = response.headers.get('content-type')
                        if content_type and not content_type.startswith('image/'):
                            st.error(f"URL {img_src} returned content type '{content_type}', not an image.")
                            st.stop()
                        image = Image.open(BytesIO(response.content))
                    else:
                        # Assuming it's a local path
                        image = Image.open(img_src)

                    if image:
                        # Draw predictions
                        annotated_image = draw_predictions_on_image(image.copy(), image_master.structured_info)

                        # Display side-by-side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption=f"Original Image ({img_src})", use_container_width=True)
                        with col2:
                            st.image(annotated_image, caption="Image with Drawn Predictions", use_container_width=True)
                    else:
                        st.error("Failed to load image object.")


                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching image from URL {img_src}: {e}")
                except FileNotFoundError:
                    st.error(f"Error: Local image file not found at path: {img_src}")
                except Exception as e:
                    st.error(f"An unexpected error occurred loading or processing image {img_src}: {e}")
            else:
                # This case should be caught earlier, but added for safety
                st.warning("No image source available to display.")


    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
    except Exception as e:
        st.error(f"An error occurred processing the CSV file: {e}")
else:
    st.info("Please upload a CSV file containing ImageMaster JSON using the sidebar.")

