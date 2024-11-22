import streamlit as st
import numpy as np
from PIL import Image
import os

# Enhanced recyclable items mapping
RECYCLABLE_ITEMS = {
    # Plastic Items with specific types
    'bottle': {'category': 'Plastic', 'color': '#3498db'},
    'plastic bottle': {'category': 'Plastic', 'color': '#3498db'},
    'water bottle': {'category': 'Plastic', 'color': '#3498db'},
    'soda bottle': {'category': 'Plastic', 'color': '#3498db'},
    'container': {'category': 'Plastic', 'color': '#3498db'},
    'plastic container': {'category': 'Plastic', 'color': '#3498db'},
    'cup': {'category': 'Plastic', 'color': '#3498db'},
    'bowl': {'category': 'Plastic', 'color': '#3498db'},
    'jug': {'category': 'Plastic', 'color': '#3498db'},
    'vase': {'category': 'Plastic', 'color': '#3498db'},
    
    # Metal Items
    'can': {'category': 'Metal', 'color': '#e74c3c'},
    'aluminum can': {'category': 'Metal', 'color': '#e74c3c'},
    'tin can': {'category': 'Metal', 'color': '#e74c3c'},
    'soda can': {'category': 'Metal', 'color': '#e74c3c'},
    'beer can': {'category': 'Metal', 'color': '#e74c3c'},
    'metal container': {'category': 'Metal', 'color': '#e74c3c'},
    
    # Glass Items
    'glass bottle': {'category': 'Glass', 'color': '#2ecc71'},
    'wine bottle': {'category': 'Glass', 'color': '#2ecc71'},
    'beer bottle': {'category': 'Glass', 'color': '#2ecc71'},
    'jar': {'category': 'Glass', 'color': '#2ecc71'},
    
    # Paper/Cardboard Items
    'box': {'category': 'Cardboard', 'color': '#f1c40f'},
    'cardboard': {'category': 'Cardboard', 'color': '#f1c40f'},
    'paper': {'category': 'Paper', 'color': '#f1c40f'},
    'book': {'category': 'Paper', 'color': '#f1c40f'},
    'newspaper': {'category': 'Paper', 'color': '#f1c40f'},
    'magazine': {'category': 'Paper', 'color': '#f1c40f'},
    # Add these to RECYCLABLE_ITEMS
    'milk jug': {'category': 'Plastic', 'color': '#3498db'},
    'detergent bottle': {'category': 'Plastic', 'color': '#3498db'},
    'soda can': {'category': 'Metal', 'color': '#e74c3c'},
    'food container': {'category': 'Plastic', 'color': '#3498db'},
    'glass jar': {'category': 'Glass', 'color': '#2ecc71'},
    'carton': {'category': 'Cardboard', 'color': '#f1c40f'},
    'cell phone': {'category': 'Electronics', 'color': '#9b59b6'},
    'keyboard': {'category': 'Electronics', 'color': '#9b59b6'},
    'scissors': {'category': 'Metal', 'color': '#e74c3c'},
    'paper bag': {'category': 'Paper', 'color': '#f1c40f'},
    'book': {'category': 'Paper', 'color': '#f1c40f'},
}

# Enhanced material mapping for better detection
MATERIAL_MAPPING = {
    # Plastic mappings
    'vase': 'plastic bottle',
    'bottle': 'plastic bottle',
    'container': 'plastic container',
    'jug': 'plastic container',
    'wine glass': 'plastic bottle',
    'cup': 'plastic container',
    'bowl': 'plastic container',
    'wine glass': 'glass bottle',
    'cell phone': 'electronics',
    'keyboard': 'electronics',
    'scissors': 'metal container',
    'paper bag': 'paper',
    'book': 'paper',
    
    # Metal mappings
    'can': 'aluminum can',
    'tin': 'tin can',
    # Add these to MATERIAL_MAPPING
    'milk jug': 'plastic container',
    'detergent': 'plastic bottle',
    'carton': 'cardboard',
    'food container': 'plastic container',
    'glass jar': 'glass bottle',
    
    # Glass mappings (only if specifically identified as glass)
    'wine bottle': 'glass bottle',
    'beer bottle': 'glass bottle'
}

def determine_material(class_name, confidence):
    # Enhanced material detection
    if any(word in class_name for word in ['bottle', 'container', 'jug']):
        if any(word in class_name for word in ['glass', 'wine', 'beer']):
            return 'glass bottle'
        return 'plastic bottle'
    elif any(word in class_name for word in ['can', 'tin', 'aluminum']):
        return 'aluminum can'
    elif any(word in class_name for word in ['box', 'carton']):
        return 'cardboard'
    return class_name

def get_recycling_instructions(category):
    instructions = {
        'Plastic': [
            "Check recycling number (1-7) on bottom",
            "Empty and rinse thoroughly",
            "Remove all caps, pumps, and labels",
            "Crush containers to save space",
            "Keep similar plastics together",
            "Check local guidelines for accepted numbers"
        ],
        'Metal': [
            "Rinse thoroughly to remove residue",
            "Remove any non-metal parts",
            "Separate aluminum from steel if required",
            "Can crush aluminum cans (if local facility allows)",
            "Remove any plastic or paper labels if possible",
            "Keep similar metals together"
        ],
        'Glass': [
            "Empty and rinse thoroughly",
            "Remove caps and lids",
            "Sort by color if required locally",
            "Don't break or crush glass",
            "Remove any non-glass parts",
            "Keep separate from other materials"
        ],
        'Paper': [
            "Keep clean and dry",
            "Remove any plastic wrapping",
            "Flatten all boxes and containers",
            "Remove any non-paper materials",
            "Bundle similar items together",
            "Don't recycle if contaminated with food/oil"
        ],
        'Cardboard': [
            "Break down all boxes flat",
            "Remove all packing materials",
            "Keep dry and clean",
            "Remove all tape and staples",
            "Bundle similar sizes together",
            "Check for recycling symbols"
        ],
        'Electronics': [
            "Remove batteries",
            "Delete personal data",
            "Keep original packaging if possible",
            "Take to certified e-waste recycler",
            "Don't break or dismantle",
            "Check for trade-in programs"
        ],
        'Mixed Materials': [
            "Separate different materials when possible",
            "Check local guidelines",
            "Clean all components",
            "Remove non-recyclable parts",
            "Consider donating if still usable",
            "Take to specialized recycling centers if needed"
        ]
    }
    return instructions.get(category, ["Check local recycling guidelines"])
def enhance_detection(class_name):
    """Better classification of detected items"""
    # Common recyclables mapping
    recycling_map = {
        'bottle': 'plastic bottle',
        'cup': 'plastic container',
        'can': 'aluminum can',
        'box': 'cardboard',
        'container': 'plastic container',
        'jar': 'glass bottle',
        'wine glass': 'glass bottle',
        'cell phone': 'electronics',
        'keyboard': 'electronics',
        'scissors': 'metal container',
        'paper bag': 'paper',
        'book': 'paper'
    }
    return recycling_map.get(class_name, class_name)
try:
    from ultralytics import YOLO
    st.success("YOLO imported successfully!")
except Exception as e:
    st.error(f"Error importing YOLO: {str(e)}")

st.title("RecycleVision - Smart Recycling Assistant")
st.write("Upload an image to identify recyclable items!")

try:
    @st.cache_resource
    def load_model():
        try:
            model = YOLO("yolov8x.pt")
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    model = load_model()

    if model:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = image.convert('RGB')
            st.image(image, caption="Original Image")

            img_array = np.array(image)

            try:
                results = model.predict(
                    source=img_array,
                    conf=0.05,        # Lower confidence threshold for detecting more items
                    iou=0.1,         # Intersection over Union threshold for overlapping items
                    max_det=100,       # Maximum number of detections per image
                    classes=[
            39,  # bottle
            40,  # wine glass
            41,  # cup
            42,  # fork
            43,  # knife
            44,  # spoon
            45,  # bowl
            46,  # banana
            47,  # apple
            49,  # orange
            50,  # broccoli
            51,  # carrot
            52,  # hot dog
            53,  # pizza
            61,  # plate
            62,  # wine glass
            63,  # cup
            64,  # fork
            65,  # knife
            66,  # spoon
            67,  # bowl
            84,  # book
            86,  # vase
            89,  # paper
            90,  # cardboard
            93   # paper bag
    ]  # Class indices for common recyclables like bottles, cups, etc.
                )
    # Add bounding box visualization
                annotated_frame = results[0].plot()
                st.image(annotated_frame, caption="Detected Items", use_container_width=True)                
                all_detections = {}
                
                for r in results:  # <- Fixed indentation
                    boxes = r.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id].lower()
                        
                        # Apply enhanced detection first
                        class_name = enhance_detection(class_name)
                        
                        # Then apply material detection
                        class_name = determine_material(class_name, confidence)
                        
                        # Finally apply material mapping
                        if class_name in MATERIAL_MAPPING:
                            class_name = MATERIAL_MAPPING[class_name]
                        
                        # Check if item is recyclable
                        if class_name in RECYCLABLE_ITEMS and confidence > 0.1:
                            category = RECYCLABLE_ITEMS[class_name]['category']
                            if category not in all_detections:
                                all_detections[category] = []
                            all_detections[category].append(class_name)

                # Display results
                if all_detections:
                    st.write("## Recyclable Items Detected")
                    
                    for category, items in all_detections.items():
                        # Count unique items
                        item_counts = {}
                        for item in items:
                            item_counts[item] = item_counts.get(item, 0) + 1
                        
                        # Display category section
                        st.markdown(f"""
                        <div style="border-left: 5px solid {RECYCLABLE_ITEMS[items[0]]['color']}; 
                             padding: 15px; margin: 15px 0; background-color: #f8f9fa; border-radius: 5px;">
                            <h3>{category}</h3>
                        """, unsafe_allow_html=True)
                        
                        # List items
                        for item, count in item_counts.items():
                            st.markdown(f"• {count} {item.replace('_', ' ').title()}{'s' if count > 1 else ''}")
                        
                        # Show recycling instructions
                        st.markdown("#### Recycling Instructions:")
                        for instruction in get_recycling_instructions(category):
                            st.markdown(f"- {instruction}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No recyclable items detected. Try adjusting the image or angle.")
                
                # Add general recycling tips
                st.write("## General Recycling Tips")
                st.info("""
                🌍 Remember:
                • Clean all items before recycling
                • Check local recycling guidelines
                • Keep different materials separated
                • When in doubt, check with your local facility
                • Consider reducing and reusing before recycling
                """)
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

except Exception as e:
    st.error(f"General error: {str(e)}")