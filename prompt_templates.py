# src/prompt_engineering/prompt_templates.py
from typing import Dict, List, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptTemplate:
    """Base class for prompt templates"""
    
    def __init__(self, template: str, required_variables: List[str]):
        self.template = template
        self.required_variables = required_variables
    
    def format(self, **kwargs) -> str:
        """Format the template with the provided variables"""
        # Check that all required variables are provided
        for var in self.required_variables:
            if var not in kwargs:
                raise ValueError(f"Missing required variable: {var}")
        
        # Format the template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable in template: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting template: {e}")

class TechnicalDocPrompts:
    """Collection of prompt templates for technical documentation"""
    
    def __init__(self):
        # Define templates for different documentation tasks
        self.templates = {
            "api_reference": PromptTemplate(
                template=(
                    "Generate comprehensive API documentation for the following function or class:\n\n"
                    "```{language}\n{code}\n```\n\n"
                    "Include the following sections:\n"
                    "- Description: A clear explanation of what this {code_type} does.\n"
                    "- Parameters: Each parameter with type, description, and whether it's required.\n"
                    "- Returns: What the {code_type} returns, with type information.\n"
                    "- Exceptions: Any exceptions that might be raised.\n"
                    "- Examples: {num_examples} example(s) showing typical usage."
                ),
                required_variables=["language", "code", "code_type", "num_examples"]
            ),
            
            "tutorial": PromptTemplate(
                template=(
                    "Create a step-by-step tutorial about {topic} with the following structure:\n\n"
                    "## Introduction\n"
                    "- Briefly explain what {topic} is and why it's useful.\n"
                    "- Mention the key concepts that will be covered.\n"
                    "- Specify any prerequisites needed.\n\n"
                    "## Steps\n"
                    "- Provide {num_steps} clear steps to accomplish the task.\n"
                    "- For each step, include explanations and code examples where applicable.\n\n"
                    "## Common Challenges and Solutions\n"
                    "- Address {num_challenges} common issues users might face.\n\n"
                    "## Next Steps\n"
                    "- Suggest related topics or advanced techniques to explore.\n"
                ),
                required_variables=["topic", "num_steps", "num_challenges"]
            ),
            
            "concept_explanation": PromptTemplate(
                template=(
                    "Explain the concept of {concept} in technical documentation format:\n\n"
                    "## What is {concept}?\n"
                    "- Provide a clear definition and explanation.\n\n"
                    "## Key Components\n"
                    "- Break down the main components or aspects of {concept}.\n\n"
                    "## How it Works\n"
                    "- Explain the underlying mechanism or process.\n"
                    "- Include technical details appropriate for {expertise_level} level.\n\n"
                    "## Use Cases\n"
                    "- Describe {num_use_cases} practical applications or scenarios.\n\n"
                    "## Related Concepts\n"
                    "- Mention closely related concepts and their relationships."
                ),
                required_variables=["concept", "expertise_level", "num_use_cases"]
            ),
            
            "troubleshooting": PromptTemplate(
                template=(
                    "Create a troubleshooting guide for {technology} problems:\n\n"
                    "## Issue: {issue}\n\n"
                    "### Symptoms\n"
                    "- List observable symptoms of this problem.\n\n"
                    "### Potential Causes\n"
                    "- Explain {num_causes} possible causes of this issue.\n\n"
                    "### Diagnostic Steps\n"
                    "- Provide a clear sequence of steps to diagnose the issue.\n"
                    "- Include commands or code snippets where applicable.\n\n"
                    "### Solutions\n"
                    "- Offer {num_solutions} solution(s) with step-by-step instructions.\n"
                    "- Indicate which solution applies to which cause.\n\n"
                    "### Prevention\n"
                    "- Suggest measures to prevent this issue in the future."
                ),
                required_variables=["technology", "issue", "num_causes", "num_solutions"]
            )
        }
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get a formatted prompt of the specified type"""
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available types: {list(self.templates.keys())}")
        
        template = self.templates[prompt_type]
        return template.format(**kwargs)
    
    def list_available_prompts(self) -> List[str]:
        """List all available prompt types"""
        return list(self.templates.keys())
    
    def get_required_variables(self, prompt_type: str) -> List[str]:
        """Get the required variables for a specific prompt type"""
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available types: {list(self.templates.keys())}")
        
        return self.templates[prompt_type].required_variables