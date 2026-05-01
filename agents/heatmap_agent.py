import os
import json
import re
import numpy as np
import cv2
from google import genai
from PIL import Image


def heatmap_agent(state: dict) -> dict:
    original_image = state.get("image")

    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")

        client = genai.Client(api_key=api_key)

        if original_image is None:
            raise ValueError("No image in state.")

        disease_label = state.get("disease_label", "Unknown")
        rgb_image = original_image.convert("RGB")

        region_prompt = (
            f"Examine this medical image. The diagnosed condition is: {disease_label}\n\n"
            "Identify the primary region affected by this condition and return ONLY a JSON "
            "bounding box using fractional coordinates (0.0 to 1.0 of image width/height):\n"
            '{"x1": 0.2, "y1": 0.3, "x2": 0.7, "y2": 0.8}\n\n'
            "x1,y1 = top-left corner; x2,y2 = bottom-right corner.\n"
            "Return ONLY the JSON object, absolutely no other text."
        )

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[region_prompt, rgb_image],
        )
        response_text = response.text.strip()

        bbox = {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}
        json_match = re.search(r'\{[^}]*"x1"[^}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                bbox = parsed
            except json.JSONDecodeError:
                pass

        x1 = max(0.0, min(float(bbox.get("x1", 0.25)), 1.0))
        y1 = max(0.0, min(float(bbox.get("y1", 0.25)), 1.0))
        x2 = max(0.0, min(float(bbox.get("x2", 0.75)), 1.0))
        y2 = max(0.0, min(float(bbox.get("y2", 0.75)), 1.0))

        if x1 >= x2:
            x1, x2 = 0.25, 0.75
        if y1 >= y2:
            y1, y2 = 0.25, 0.75

        img_array = np.array(rgb_image, dtype=np.float32)
        h, w = img_array.shape[:2]

        cx = int((x1 + x2) / 2 * w)
        cy = int((y1 + y2) / 2 * h)
        sigma_x = max(int((x2 - x1) * w / 3), 30)
        sigma_y = max(int((y2 - y1) * h / 3), 30)

        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        heatmap = np.exp(
            -((xx - cx) ** 2 / (2 * sigma_x ** 2) + (yy - cy) ** 2 / (2 * sigma_y ** 2))
        )
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        blended = (img_array * 0.55 + heatmap_rgb * 0.45).clip(0, 255).astype(np.uint8)

        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(blended, pt1, pt2, (255, 0, 0), 3)

        heatmap_pil = Image.fromarray(blended)
        return {"heatmap_image": heatmap_pil}

    except Exception as e:
        return {
            "heatmap_image": original_image,
            "error": f"Heatmap agent error: {str(e)}",
        }
