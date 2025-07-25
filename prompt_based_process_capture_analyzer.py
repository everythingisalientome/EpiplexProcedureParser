import pandas as pd
import json
import openai
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SinglePromptProcessor:
    """Simple processor that uses one comprehensive prompt to analyze CSV and generate JSON"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model_name = model_name
    
    def get_comprehensive_prompt(self) -> str:
        """Returns the comprehensive prompt for processing process capture CSV"""
        return """
You are an expert automation engineer tasked with analyzing process capture CSV data and generating automation scripts. 

TASK: Convert the provided CSV data into a JSON array of automation steps following the exact format specified below.

ANALYSIS INSTRUCTIONS:

1. **APPLICATION TYPE CLASSIFICATION**:
   - If URL or Domain columns have values → "web"
   - If "mainframe" appears in Application Category → "mainframe" 
   - Otherwise → "desktop"
   - Use ExeName, Exe Description, Application Category, Application, Work Category for context

2. **SELECTOR GENERATION**:
   For WEB applications:
   - Priority: XPath > FieldName > Role > generate new selector
   - Generate Selenium-compatible selectors (XPath or CSS)
   - Examples: "//button[text()='Submit']", "//input[@name='username']", "#login-button"
   
   For DESKTOP/MAINFRAME applications:
   - Use WindowName, FieldName, Role, UniqueID, ScreenName
   - Generate PyAutoGUI or accessibility-based selectors
   - Examples: "pyautogui.accessibility(name='Submit', role='button')", "opencv.template_match('button.png')"
   
   When Capture Image Path exists:
   - Reference the image for better selector generation
   - Use format: "opencv.template_match('[image_path]')" or "pyautogui.locateOnScreen('[image_path]')"

3. **ACTION TYPE MAPPING**:
   - Event "click" → "click"
   - Event "type"/"input" → "typeInto" 
   - Event "select" → "select"
   - Event "scroll" → "scroll"
   - Event "drag" → "drag"
   - FieldType "button" → "click"
   - FieldType "textbox"/"input" → "typeInto"
   - FieldType "dropdown" → "select"

4. **SPECIAL KEYS HANDLING**:
   Parse SpecialKeyWithData column:
   - "CTRL+V" → action_type: "hotkey", special_keys: ["ctrl", "v"]
   - "SHIFT+ENTER" → action_type: "hotkey", special_keys: ["shift", "enter"]  
   - "ENTER" → action_type: "keyPress", special_keys: ["enter"]
   - "TAB" → action_type: "keyPress", special_keys: ["tab"]
   - Any combination → parse accordingly

5. **UNICODE CLEANING**:
   Clean all text data by:
   - Replace smart quotes (" ") with regular quotes (" ")
   - Replace em/en dashes (– —) with hyphens (-)
   - Replace ellipsis (…) with three dots (...)
   - Replace bullets (•) with asterisk (*)
   - Remove trademark/copyright symbols (® © ™)
   - Keep only ASCII characters for selectors
   - For URLs: preserve structure, only replace problematic characters

6. **SEQUENCING**:
   - Use row order as step sequence (1, 2, 3, ...)
   - Each CSV row = one automation step
   - Maintain chronological order from CSV

7. **URL/EXE EXTRACTION**:
   - application_url: Use URL column value
   - exe_path: Use ExeName column value
   - If both exist, prioritize based on application_type

IGNORE THESE COLUMNS (don't use for processing):
RepeatedCount, UTCStartTime, UTCEndTime, StartTime, EndTime, Duration, ActionDuration, ThinkDuration, IdleDuration, ExcludeDuration, SystemLockDuration

OUTPUT FORMAT:
Return ONLY a valid JSON array with this exact structure:
[
   {
       "application_type": "web|desktop|mainframe",
       "application_url": "https://example.com" or null,
       "exe_path": "C:\\path\\to\\app.exe" or null,
       "field_name": "field_name_value" or null,
       "action_description": "description from Sentence column",
       "selector": "generated_selector_here",
       "event_action_type": "click|typeInto|select|scroll|drag|hotkey|keyPress",
       "step_sequence": "1",
       "special_keys": ["key1", "key2"] // only include if SpecialKeyWithData has values
   }
]

IMPORTANT RULES:
- Return ONLY the JSON array, no explanations or additional text
- Include "special_keys" field ONLY when SpecialKeyWithData has values
- Clean all unicode characters as specified
- Generate the most reliable selector for each application type
- Use Sentence column as action_description
- Ensure all selectors are properly formatted for their automation tool
- Handle missing data gracefully (use null for missing values)
- Maintain exact step sequence from CSV row order

Now analyze the CSV data below and generate the JSON output:
"""

    def process_csv_with_single_prompt(self, csv_path: str) -> List[Dict[str, Any]]:
        """Process CSV using single comprehensive prompt"""
        logger.info(f"Loading CSV: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Convert DataFrame to string representation for the prompt
        csv_string = df.to_string(index=True, max_rows=None)
        
        # Combine prompt with CSV data
        full_prompt = self.get_comprehensive_prompt() + "\n\nCSV DATA:\n" + csv_string
        
        logger.info(f"Processing {len(df)} rows with LLM...")
        
        try:
            # Send to LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert automation engineer. Follow the instructions precisely and return only valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": full_prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=16000  # Sufficient for large JSON outputs
            )
            
            # Extract JSON from response
            json_output = response.choices[0].message.content.strip()
            
            # Clean up response in case there's extra text
            json_output = self._extract_json_from_response(json_output)
            
            # Parse and validate JSON
            result = json.loads(json_output)
            
            logger.info(f"Successfully generated {len(result)} automation steps")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {json_output}")
            raise
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON array from LLM response, handling potential extra text"""
        response = response.strip()
        
        # Find JSON array boundaries
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx + 1]
        
        # If no clear boundaries, return as-is and let JSON parser handle the error
        return response
    
    def process_csv_chunked(self, csv_path: str, chunk_size: int = 50) -> List[Dict[str, Any]]:
        """Process large CSV in chunks to handle token limits"""
        logger.info(f"Processing CSV in chunks of {chunk_size} rows")
        
        df = pd.read_csv(csv_path)
        all_results = []
        
        # Process in chunks
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size].copy()
            
            # Adjust step sequence to maintain global order
            chunk_df.index = range(i, i + len(chunk_df))
            
            logger.info(f"Processing chunk {i//chunk_size + 1}: rows {i+1} to {min(i+chunk_size, len(df))}")
            
            # Convert chunk to string
            csv_string = chunk_df.to_string(index=True, max_rows=None)
            full_prompt = self.get_comprehensive_prompt() + "\n\nCSV DATA:\n" + csv_string
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert automation engineer. Follow the instructions precisely and return only valid JSON."
                        },
                        {
                            "role": "user", 
                            "content": full_prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=16000
                )
                
                json_output = response.choices[0].message.content.strip()
                json_output = self._extract_json_from_response(json_output)
                chunk_result = json.loads(json_output)
                
                all_results.extend(chunk_result)
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i//chunk_size + 1}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(all_results)} total steps")
        return all_results

def main():
    """Example usage"""
    
    # Initialize processor
    processor = SinglePromptProcessor(
        openai_api_key="your-openai-api-key",
        model_name="gpt-4"  # or "gpt-3.5-turbo" for faster/cheaper processing
    )
    
    try:
        # Process CSV (choose one method)
        
        # Method 1: Single prompt (for smaller CSVs < 100 rows)
        results = processor.process_csv_with_single_prompt("process_capture_data.csv")
        
        # Method 2: Chunked processing (for larger CSVs)
        # results = processor.process_csv_chunked("process_capture_data.csv", chunk_size=50)
        
        # Save results
        output_file = "automation_steps.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Processing complete! Results saved to {output_file}")
        print(f"📊 Generated {len(results)} automation steps")
        
        # Preview first few steps
        print("\n🔍 Preview of first 3 steps:")
        for i, step in enumerate(results[:3]):
            print(f"\nStep {i+1}:")
            print(json.dumps(step, indent=2))
    
    except Exception as e:
        print(f"❌ Processing failed: {e}")

if __name__ == "__main__":
    main()