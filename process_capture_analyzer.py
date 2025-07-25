import pandas as pd
import json
import re
import unicodedata
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import openai
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor
import cv2
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ApplicationType(Enum):
    WEB = "web"
    DESKTOP = "desktop"
    MAINFRAME = "mainframe"

@dataclass
class ProcessStep:
    """Represents a single process step"""
    row_index: int
    application_type: str
    application_url: Optional[str]
    exe_path: Optional[str]
    field_name: Optional[str]
    action_description: str
    selector: str
    event_action_type: str
    step_sequence: str
    confidence_score: float = 0.0
    special_keys: Optional[List[str]] = None

@dataclass
class ProcessState:
    """State object for LangGraph"""
    csv_data: pd.DataFrame
    current_row: Dict[str, Any]
    current_step: ProcessStep
    processed_steps: List[ProcessStep]
    validation_results: Dict[str, Any]
    error_messages: List[str]

class UnicodeCleaningStrategy(Enum):
    """Unicode cleaning strategies"""
    REMOVE_ALL = "remove_all"  # Remove all non-ASCII
    NORMALIZE_KEEP_ALPHANUMERIC = "normalize_alphanumeric"  # Keep letters, numbers, basic symbols
    SMART_REPLACE = "smart_replace"  # Replace common unicode with ASCII equivalents

class ProcessCaptureAnalyzer:
    """Main analyzer class using LangGraph"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4", 
                 unicode_strategy: UnicodeCleaningStrategy = UnicodeCleaningStrategy.SMART_REPLACE):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.unicode_strategy = unicode_strategy
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ProcessState)
        
        # Add nodes
        workflow.add_node("load_data", self.load_data_node)
        workflow.add_node("classify_application", self.classify_application_node)
        workflow.add_node("generate_web_selector", self.generate_web_selector_node)
        workflow.add_node("generate_desktop_selector", self.generate_desktop_selector_node)
        workflow.add_node("map_action_type", self.map_action_type_node)
        workflow.add_node("clean_unicode", self.clean_unicode_node)  # New unicode cleaning step
        workflow.add_node("validate_step", self.validate_step_node)
        workflow.add_node("regenerate_selector", self.regenerate_selector_node)
        workflow.add_node("finalize_step", self.finalize_step_node)
        
        # Set entry point
        workflow.set_entry_point("load_data")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "classify_application",
            self._route_by_application_type,
            {
                "web": "generate_web_selector",
                "desktop": "generate_desktop_selector",
                "mainframe": "generate_desktop_selector"
            }
        )
        
        # Connect remaining edges
        workflow.add_edge("load_data", "classify_application")
        workflow.add_edge("generate_web_selector", "map_action_type")
        workflow.add_edge("generate_desktop_selector", "map_action_type")
        workflow.add_edge("map_action_type", "clean_unicode")  # Add unicode cleaning step
        workflow.add_edge("clean_unicode", "validate_step")
        
        workflow.add_conditional_edges(
            "validate_step",
            self._route_validation_result,
            {
                "regenerate": "regenerate_selector",
                "finalize": "finalize_step"
            }
        )
        
        workflow.add_edge("regenerate_selector", "clean_unicode")  # Clean unicode after regeneration too
        workflow.add_edge("finalize_step", END)
        
        return workflow.compile()
    
    def _route_by_application_type(self, state: ProcessState) -> str:
        """Route based on application type"""
        return state
    
    def clean_unicode_node(self, state: ProcessState) -> ProcessState:
        """Clean unicode characters from selectors and relevant fields"""
        step = state.current_step
        
        # Clean selector
        step.selector = self._clean_unicode_string(step.selector)
        
        # Clean field name if present
        if step.field_name:
            step.field_name = self._clean_unicode_string(step.field_name)
        
        # Clean action description
        step.action_description = self._clean_unicode_string(step.action_description)
        
        # Clean application URL if present
        if step.application_url:
            step.application_url = self._clean_unicode_url(step.application_url)
        
        # Clean exe path if present
        if step.exe_path:
            step.exe_path = self._clean_unicode_string(step.exe_path)
        
        logger.info(f"Unicode cleaning completed for step {step.step_sequence}")
        return state
    
    def _clean_unicode_string(self, text: str) -> str:
        """Clean unicode characters from a string based on strategy"""
        if not text or not isinstance(text, str):
            return text or ""
        
        if self.unicode_strategy == UnicodeCleaningStrategy.REMOVE_ALL:
            # Remove all non-ASCII characters
            return ''.join(char for char in text if ord(char) < 128)
        
        elif self.unicode_strategy == UnicodeCleaningStrategy.NORMALIZE_KEEP_ALPHANUMERIC:
            # Normalize unicode and keep only alphanumeric + basic symbols
            normalized = unicodedata.normalize('NFKD', text)
            # Keep letters, numbers, and common selector symbols
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                              '[](){}@#.*-_=:/"\'`~!$%^&+|\\<>,? ')
            return ''.join(char for char in normalized if char in allowed_chars)
        
        elif self.unicode_strategy == UnicodeCleaningStrategy.SMART_REPLACE:
            # Smart replacement of common unicode characters
            return self._smart_unicode_replacement(text)
        
        return text
    
    def _clean_unicode_url(self, url: str) -> str:
        """Special cleaning for URLs to preserve functionality"""
        if not url or not isinstance(url, str):
            return url or ""
        
        # For URLs, we're more conservative - only replace problematic characters
        # Keep URL structure intact
        url_cleaned = url
        
        # Common unicode replacements that don't break URLs
        unicode_replacements = {
            '"': '"',  # Smart quotes
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '…': '...',  # Ellipsis
            '®': '',   # Registered trademark
            '™': '',   # Trademark
            '©': '',   # Copyright
        }
        
        for unicode_char, ascii_replacement in unicode_replacements.items():
            url_cleaned = url_cleaned.replace(unicode_char, ascii_replacement)
        
        # Remove any remaining non-ASCII characters from URL
        url_cleaned = ''.join(char for char in url_cleaned if ord(char) < 128)
        
        return url_cleaned
    
    def _smart_unicode_replacement(self, text: str) -> str:
        """Smart replacement of unicode characters with ASCII equivalents"""
        # Normalize unicode first
        text = unicodedata.normalize('NFKD', text)
        
        # Common unicode to ASCII mappings for UI elements
        unicode_replacements = {
            # Quotes
            '"': '"', '"': '"', ''': "'", ''': "'",
            # Dashes
            '–': '-', '—': '-', '−': '-',
            # Spaces and separators
            '\u00a0': ' ',  # Non-breaking space
            '\u2000': ' ',  # En quad
            '\u2001': ' ',  # Em quad
            '\u2002': ' ',  # En space
            '\u2003': ' ',  # Em space
            '\u2004': ' ',  # Three-per-em space
            '\u2005': ' ',  # Four-per-em space
            '\u2006': ' ',  # Six-per-em space
            '\u2007': ' ',  # Figure space
            '\u2008': ' ',  # Punctuation space
            '\u2009': ' ',  # Thin space
            '\u200a': ' ',  # Hair space
            '\u202f': ' ',  # Narrow no-break space
            '\u205f': ' ',  # Medium mathematical space
            '\u3000': ' ',  # Ideographic space
            # Dots and bullets
            '•': '*', '·': '.', '…': '...',
            # Symbols
            '®': '', '™': '', '©': '',
            # Mathematical symbols
            '×': 'x', '÷': '/',
            # Arrows (often in UI)
            '→': '->', '←': '<-', '↑': '^', '↓': 'v',
            '⇒': '=>', '⇐': '<=',
        }
        
        # Apply replacements
        for unicode_char, ascii_replacement in unicode_replacements.items():
            text = text.replace(unicode_char, ascii_replacement)
        
        # Remove accents and diacritics, keep base characters
        text = ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )
        
        # Final fallback: remove any remaining non-ASCII characters
        # but preserve common selector symbols
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                          '[](){}@#.*-_=:/"\'`~!$%^&+|\\<>,? ')
        text = ''.join(char for char in text if ord(char) < 128 and char in allowed_chars)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.current_step.application_type
    
    def _route_validation_result(self, state: ProcessState) -> str:
        """Route based on validation confidence score"""
        if state.current_step.confidence_score < 0.8:
            return "regenerate"
        return "finalize"
    
    def load_data_node(self, state: ProcessState) -> ProcessState:
        """Load and prepare CSV data"""
        logger.info("Loading CSV data...")
        
        # Sort by timestamp to maintain sequence
        if 'UTCStartTime' in state.csv_data.columns:
            state.csv_data = state.csv_data.sort_values('UTCStartTime')
        elif 'StartTime' in state.csv_data.columns:
            state.csv_data = state.csv_data.sort_values('StartTime')
        
        return state
    
    def classify_application_node(self, state: ProcessState) -> ProcessState:
        """Classify application type using LLM"""
        row = state.current_row
        
        # Primary classification logic
        if pd.notna(row.get('URL')) or pd.notna(row.get('Domain')):
            app_type = ApplicationType.WEB.value
        elif 'mainframe' in str(row.get('Application Category', '')).lower():
            app_type = ApplicationType.MAINFRAME.value
        else:
            app_type = ApplicationType.DESKTOP.value
        
        # Use LLM for ambiguous cases
        if self._is_classification_ambiguous(row):
            app_type = self._llm_classify_application(row)
        
        # Initialize ProcessStep
        state.current_step = ProcessStep(
            row_index=row.name,
            application_type=app_type,
            application_url=row.get('URL'),
            exe_path=row.get('ExeName'),
            field_name=row.get('FieldName'),
            action_description=row.get('Sentence', ''),
            selector="",  # To be generated
            event_action_type="",  # To be mapped
            step_sequence=str(row.name + 1)
        )
        
        logger.info(f"Classified application type: {app_type}")
        return state
    
    def _is_classification_ambiguous(self, row: Dict) -> bool:
        """Check if application classification needs LLM assistance"""
        url_empty = pd.isna(row.get('URL'))
        exe_empty = pd.isna(row.get('ExeName'))
        category_generic = str(row.get('Application Category', '')).lower() in ['', 'other', 'unknown']
        
        return url_empty and exe_empty and category_generic
    
    def _llm_classify_application(self, row: Dict) -> str:
        """Use LLM to classify ambiguous application types"""
        prompt = f"""
        Analyze the following application data and classify it as 'web', 'desktop', or 'mainframe':
        
        ExeName: {row.get('ExeName', 'N/A')}
        Exe Description: {row.get('Exe Description', 'N/A')}
        Application Category: {row.get('Application Category', 'N/A')}
        Application: {row.get('Application', 'N/A')}
        Work Category: {row.get('Work Category', 'N/A')}
        WindowName: {row.get('WindowName', 'N/A')}
        Sentence: {row.get('Sentence', 'N/A')}
        URL: {row.get('URL', 'N/A')}
        Domain: {row.get('Domain', 'N/A')}
        
        Respond with only: 'web', 'desktop', or 'mainframe'
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result = response.choices[0].message.content.strip().lower()
            return result if result in ['web', 'desktop', 'mainframe'] else 'desktop'
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return ApplicationType.DESKTOP.value
    
    def generate_web_selector_node(self, state: ProcessState) -> ProcessState:
        """Generate Selenium selectors for web applications"""
        row = state.current_row
        
        # Priority order: XPath -> FieldName -> Role -> LLM generation
        selector = ""
        
        if pd.notna(row.get('XPath')):
            selector = row['XPath']
        elif pd.notna(row.get('FieldName')):
            field_name = row['FieldName']
            field_type = row.get('FieldType', '').lower()
            
            if 'button' in field_type:
                selector = f"//button[contains(@name, '{field_name}') or contains(text(), '{field_name}')]"
            elif 'input' in field_type or 'text' in field_type:
                selector = f"//input[@name='{field_name}' or @id='{field_name}']"
            else:
                selector = f"//*[@name='{field_name}' or @id='{field_name}']"
        
        elif pd.notna(row.get('Role')):
            role = row['Role']
            selector = f"//*[@role='{role}']"
        
        # Fallback to LLM generation
        if not selector:
            selector = self._llm_generate_web_selector(row)
        
        state.current_step.selector = selector
        logger.info(f"Generated web selector: {selector}")
        return state
    
    def _llm_generate_web_selector(self, row: Dict) -> str:
        """Use LLM to generate web selector when standard methods fail"""
        prompt = f"""
        Generate a Selenium XPath or CSS selector for the following web element:
        
        WindowName: {row.get('WindowName', 'N/A')}
        FieldName: {row.get('FieldName', 'N/A')}
        FieldType: {row.get('FieldType', 'N/A')}
        Title: {row.get('Title', 'N/A')}
        Header: {row.get('Header', 'N/A')}
        Role: {row.get('Role', 'N/A')}
        Sentence: {row.get('Sentence', 'N/A')}
        
        Provide only the selector (XPath preferred). No explanation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM web selector generation failed: {e}")
            return f"//input[@name='{row.get('FieldName', 'unknown')}']"
    
    def generate_desktop_selector_node(self, state: ProcessState) -> ProcessState:
        """Generate PyAutoGUI/OpenCV selectors for desktop applications"""
        row = state.current_row
        
        # Build accessibility-based selector first
        selector_parts = []
        
        if pd.notna(row.get('WindowName')):
            selector_parts.append(f"window='{row['WindowName']}'")
        
        if pd.notna(row.get('FieldName')):
            selector_parts.append(f"name='{row['FieldName']}'")
        
        if pd.notna(row.get('Role')):
            selector_parts.append(f"role='{row['Role']}'")
        
        if selector_parts:
            selector = f"pyautogui.accessibility({', '.join(selector_parts)})"
        else:
            # Fallback to image-based selector
            image_path = row.get('Capture Image Path')
            if pd.notna(image_path) and Path(image_path).exists():
                selector = f"opencv.template_match('{image_path}')"
            else:
                # LLM generation for desktop
                selector = self._llm_generate_desktop_selector(row)
        
        state.current_step.selector = selector
        logger.info(f"Generated desktop selector: {selector}")
        return state
    
    def _llm_generate_desktop_selector(self, row: Dict) -> str:
        """Use LLM to generate desktop selector"""
        prompt = f"""
        Generate a PyAutoGUI or accessibility-based selector for this desktop application element:
        
        WindowName: {row.get('WindowName', 'N/A')}
        FieldName: {row.get('FieldName', 'N/A')}
        FieldType: {row.get('FieldType', 'N/A')}
        UniqueID: {row.get('UniqueID', 'N/A')}
        ScreenName: {row.get('ScreenName', 'N/A')}
        Sentence: {row.get('Sentence', 'N/A')}
        Role: {row.get('Role', 'N/A')}
        
        Format as: pyautogui.method() or opencv.method()
        Provide only the selector code. No explanation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM desktop selector generation failed: {e}")
            return f"pyautogui.locateOnScreen('{row.get('FieldName', 'element')}.png')"
    
    def map_action_type_node(self, state: ProcessState) -> ProcessState:
        """Map Event and SpecialKeyWithData to action types"""
        row = state.current_row
        event = str(row.get('Event', '')).lower()
        special_keys = row.get('SpecialKeyWithData', '')
        field_type = str(row.get('FieldType', '')).lower()
        
        # Handle special key combinations
        if pd.notna(special_keys) and special_keys.strip():
            action_type, keys = self._parse_special_keys(special_keys)
            state.current_step.event_action_type = action_type
            state.current_step.special_keys = keys
        
        # Map based on event type
        elif 'click' in event:
            state.current_step.event_action_type = "click"
        elif 'type' in event or 'input' in event:
            state.current_step.event_action_type = "typeInto"
        elif 'select' in event:
            state.current_step.event_action_type = "select"
        elif 'scroll' in event:
            state.current_step.event_action_type = "scroll"
        elif 'drag' in event:
            state.current_step.event_action_type = "drag"
        
        # Fallback based on field type
        elif 'button' in field_type:
            state.current_step.event_action_type = "click"
        elif 'text' in field_type or 'input' in field_type:
            state.current_step.event_action_type = "typeInto"
        elif 'dropdown' in field_type or 'select' in field_type:
            state.current_step.event_action_type = "select"
        else:
            state.current_step.event_action_type = "click"  # Default
        
        logger.info(f"Mapped action type: {state.current_step.event_action_type}")
        return state
    
    def _parse_special_keys(self, special_keys_data: str) -> Tuple[str, List[str]]:
        """Parse special key combinations"""
        special_keys_data = special_keys_data.strip()
        
        # Common patterns
        if 'ctrl+v' in special_keys_data.lower():
            return "hotkey", ["ctrl", "v"]
        elif 'ctrl+c' in special_keys_data.lower():
            return "hotkey", ["ctrl", "c"]
        elif 'shift+enter' in special_keys_data.lower():
            return "hotkey", ["shift", "enter"]
        elif 'alt+tab' in special_keys_data.lower():
            return "hotkey", ["alt", "tab"]
        elif 'enter' in special_keys_data.lower():
            return "keyPress", ["enter"]
        elif 'tab' in special_keys_data.lower():
            return "keyPress", ["tab"]
        elif 'escape' in special_keys_data.lower():
            return "keyPress", ["escape"]
        else:
            # Parse custom combinations
            keys = re.split(r'[+\s]+', special_keys_data.lower())
            if len(keys) > 1:
                return "hotkey", keys
            else:
                return "keyPress", keys
    
    def validate_step_node(self, state: ProcessState) -> ProcessState:
        """Validate selector and action using LLM reflection"""
        step = state.current_step
        row = state.current_row
        
        validation_prompt = f"""
        Validate the following automation step:
        
        Action Description: {step.action_description}
        Generated Selector: {step.selector}
        Action Type: {step.event_action_type}
        Application Type: {step.application_type}
        
        Context from capture:
        Field Name: {row.get('FieldName', 'N/A')}
        Field Type: {row.get('FieldType', 'N/A')}
        Event: {row.get('Event', 'N/A')}
        
        Rate this automation step on a scale of 0-100 based on:
        1. Does the selector accurately target the described element?
        2. Does the action type match the user behavior?
        3. Will this step work reliably in automation?
        
        Respond with only a number (0-100).
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            confidence_score = float(re.search(r'\d+', score_text).group()) / 100.0
            step.confidence_score = confidence_score
            
            logger.info(f"Validation confidence score: {confidence_score}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            step.confidence_score = 0.5  # Default to regenerate
        
        return state
    
    def regenerate_selector_node(self, state: ProcessState) -> ProcessState:
        """Regenerate selector using LLM with full context"""
        step = state.current_step
        row = state.current_row
        
        regeneration_prompt = f"""
        The previous selector failed validation. Generate a better selector:
        
        Previous Selector: {step.selector}
        Application Type: {step.application_type}
        Action Description: {step.action_description}
        
        Full Context:
        WindowName: {row.get('WindowName', 'N/A')}
        FieldName: {row.get('FieldName', 'N/A')}
        FieldType: {row.get('FieldType', 'N/A')}
        Title: {row.get('Title', 'N/A')}
        Header: {row.get('Header', 'N/A')}
        Role: {row.get('Role', 'N/A')}
        XPath: {row.get('XPath', 'N/A')}
        UniqueID: {row.get('UniqueID', 'N/A')}
        ScreenName: {row.get('ScreenName', 'N/A')}
        Sentence: {row.get('Sentence', 'N/A')}
        
        Generate a more robust selector for {step.application_type} automation.
        Provide only the selector code.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": regeneration_prompt}],
                temperature=0.3
            )
            
            new_selector = response.choices[0].message.content.strip()
            step.selector = new_selector
            logger.info(f"Regenerated selector: {new_selector}")
            
        except Exception as e:
            logger.error(f"Selector regeneration failed: {e}")
        
        return state
    
    def finalize_step_node(self, state: ProcessState) -> ProcessState:
        """Finalize and add step to processed steps"""
        state.processed_steps.append(state.current_step)
        logger.info(f"Finalized step {state.current_step.step_sequence}")
        return state
    
    def process_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Main method to process the entire CSV"""
        logger.info(f"Starting CSV processing: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Initialize state
        initial_state = ProcessState(
            csv_data=df,
            current_row={},
            current_step=None,
            processed_steps=[],
            validation_results={},
            error_messages=[]
        )
        
        # Process each row
        for index, row in df.iterrows():
            try:
                # Update state for current row
                current_state = ProcessState(
                    csv_data=df,
                    current_row=row.to_dict(),
                    current_step=None,
                    processed_steps=initial_state.processed_steps,
                    validation_results={},
                    error_messages=initial_state.error_messages
                )
                
                # Run the graph for this row
                result_state = self.graph.invoke(current_state)
                
                # Update processed steps
                initial_state.processed_steps = result_state.processed_steps
                
            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
                initial_state.error_messages.append(f"Row {index}: {str(e)}")
        
        # Convert to JSON format
        return self._convert_to_json_output(initial_state.processed_steps)
    
    def _convert_to_json_output(self, steps: List[ProcessStep]) -> List[Dict[str, Any]]:
        """Convert ProcessStep objects to JSON format"""
        json_output = []
        
        for step in steps:
            step_dict = {
                "application_type": step.application_type,
                "application_url": step.application_url,
                "exe_path": step.exe_path,
                "field_name": step.field_name,
                "action_description": step.action_description,
                "selector": step.selector,
                "event_action_type": step.event_action_type,
                "step_sequence": step.step_sequence
            }
            
            # Add special keys if present
            if step.special_keys:
                step_dict["special_keys"] = step.special_keys
            
            json_output.append(step_dict)
        
        return json_output

# Usage Example
def main():
    """Example usage of the ProcessCaptureAnalyzer"""
    
    # Initialize analyzer
    analyzer = ProcessCaptureAnalyzer(
        openai_api_key="your-openai-api-key",
        model_name="gpt-4",
        unicode_strategy=UnicodeCleaningStrategy.SMART_REPLACE  # Choose cleaning strategy
    )
    
    # Process CSV file
    try:
        results = analyzer.process_csv("process_capture_data.csv")
        
        # Save results
        output_file = "automation_steps.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Processing complete! Results saved to {output_file}")
        print(f"Generated {len(results)} automation steps")
        
        # Print first few steps for preview
        for i, step in enumerate(results[:3]):
            print(f"\nStep {i+1}:")
            print(json.dumps(step, indent=2))
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main()