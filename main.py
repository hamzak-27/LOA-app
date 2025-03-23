import os
import json
import datetime
from typing import Dict, Any, List, Optional
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class LOAGenerator:
    """
    A class for generating Letters of Authorization (LOAs) for outdoor advertising
    based on specific parameters using OpenAI's API.
    """
    
    def __init__(self):
        self.model = "gpt-4"  # Can be changed to gpt-3.5-turbo for lower cost
        self.conversation_history = []
        self.current_loa = None
        
    def _create_system_prompt(self) -> str:
        """
        Creates the system prompt that instructs the model on how to generate LOAs.
        
        Returns:
            str: The system prompt
        """
        return """
        You are an expert in creating Letters of Authorization (LOAs) for outdoor advertising agencies. 
        Your task is to generate professional, legally-sound LOAs based on the parameters provided.
        
        Follow these guidelines:
        1. Use formal business language appropriate for official documents
        2. Structure the LOA with proper sections including reference numbers, dates, recipient details, subject line, main body, and signatory information
        3. Include all necessary legal clauses regarding installation, maintenance, payments, and liability
        4. Format dates as DD.MM.YYYY
        5. Make the content specific to the scenario provided
        6. Ensure payment terms and conditions are clearly stated
        7. Include appropriate references to any tenders or previous communications when provided
        
        Return only the plain text content of the LOA without any explanations or additional formatting.
        """
    
    def _construct_loa_prompt(self, params: Dict[str, Any]) -> str:
        """
        Constructs a prompt for the model to generate an LOA based on the provided parameters.
        
        Args:
            params: Dictionary of parameters for LOA generation
            
        Returns:
            str: The constructed prompt
        """
        # Extract and format recipient address
        address_parts = []
        if params.get("company_name"):
            address_parts.append(params["company_name"])
        if params.get("address_line1"):
            address_parts.append(params["address_line1"])
        if params.get("address_line2"):
            address_parts.append(params["address_line2"])
        if params.get("address_line3"):
            address_parts.append(params["address_line3"])
        
        city_pincode = ""
        if params.get("city"):
            city_pincode += params["city"]
        if params.get("pincode"):
            city_pincode += f" - {params['pincode']}"
        if city_pincode:
            address_parts.append(city_pincode)
            
        recipient_address = "\n".join(address_parts)
        
        # Format date
        date_str = ""
        if isinstance(params.get("date"), datetime.date):
            date_str = params["date"].strftime("%d.%m.%Y")
        elif params.get("date"):
            date_str = params["date"]
        else:
            date_str = datetime.date.today().strftime("%d.%m.%Y")
        
        # Construct the prompt
        prompt = f"""
        Generate a Letter of Authorization (LOA) with the following details:
        
        Reference Number: {params.get('reference_number', 'LOA/' + datetime.date.today().strftime('%Y/%m/%d'))}
        Date: {date_str}
        
        Recipient:
        {recipient_address}
        
        """
        
        # Add contact person if provided
        if params.get("contact_person"):
            prompt += f"Kind attention: {params['contact_person']}\n"
            
        if params.get("contact_email"):
            prompt += f"Email: {params['contact_email']}\n"
            
        if params.get("contact_phone"):
            prompt += f"Phone: {params['contact_phone']}\n"
            
        # Add scenario information
        prompt += f"""
        Scenario: {params.get('scenario', 'outdoor advertising')}
        
        Subject: LOA for {params.get('scenario_description', 'Outdoor Advertisement')} at {params.get('location', '[Location]')}
        
        Duration: {params.get('duration', '5')} years
        """
        
        # Add specifics based on the scenario
        if params.get("size"):
            prompt += f"Size of Advertising Space: {params['size']}\n"
            
        if params.get("payment_type") and params.get("payment_amount"):
            prompt += f"""
            Payment Details:
            - Type: {params['payment_type']}
            - Amount: {params['payment_amount']} per {params.get('payment_unit', 'square foot')}
            """
            
            if params.get("annual_increase"):
                prompt += f"- Annual Increase: {params['annual_increase']}%\n"
                
        # Add any additional terms
        if params.get("additional_terms"):
            prompt += f"""
            Additional Terms:
            {params['additional_terms']}
            """
            
        # Add signatory information
        prompt += f"""
        Signatory:
        Name: {params.get('signatory_name', '[Signatory Name]')}
        Position: {params.get('signatory_position', '[Position]')}
        Organization: {params.get('organization', '[Organization]')}
        """
        
        # Add any specific requirements
        if params.get("special_requirements"):
            prompt += f"""
            Special Requirements:
            {params['special_requirements']}
            """
            
        # Add reference to sample LOAs for style guidance
        prompt += """
        Base the style and format on typical outdoor advertising LOAs which include:
        1. A formal header with reference number and date
        2. Clear recipient information
        3. A specific subject line stating the purpose
        4. An introduction referencing any tender or previous communication
        5. A main section clearly authorizing the advertising and stating terms
        6. Detailed conditions including payment terms, responsibilities, and operational requirements
        7. Standard legal clauses covering liability, termination, and compliance
        8. A formal closing with signatory information
        """
        
        
        return prompt
    
    def generate_loa(self, params: Dict[str, Any]) -> str:
        """
        Generates an LOA based on the provided parameters.
        
        Args:
            params: Dictionary of parameters for LOA generation
            
        Returns:
            str: The generated LOA text
        """
        # Create messages for the API call
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": self._construct_loa_prompt(params)}
        ]
        
        # Add conversation history if available
        if self.conversation_history:
            messages.extend(self.conversation_history)
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,  # Lower temperature for more consistent outputs
            max_tokens=2500
        )
        
        # Get the generated LOA content
        loa_content = response.choices[0].message.content
        
        # Store the conversation
        self.conversation_history = [
            {"role": "user", "content": self._construct_loa_prompt(params)},
            {"role": "assistant", "content": loa_content}
        ]
        
        # Store the current LOA
        self.current_loa = loa_content
        
        return loa_content
    
    def edit_loa(self, edit_request: str) -> str:
        """
        Edits the previously generated LOA based on the edit request.
        
        Args:
            edit_request: Description of the edits to make
            
        Returns:
            str: The edited LOA text
        """
        if not self.current_loa:
            return "No LOA has been generated yet. Please generate an LOA first."
        
        # Create the edit request message
        edit_message = {
            "role": "user", 
            "content": f"""
            Edit the LOA according to the following request:
            
            {edit_request}
            
            Return the complete edited LOA.
            """
        }
        
        # Add the edit request to the conversation history
        self.conversation_history.append(edit_message)
        
        # Create messages for the API call
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=2500
        )
        
        # Get the edited LOA content
        edited_loa = response.choices[0].message.content
        
        # Update the conversation history
        self.conversation_history.append({"role": "assistant", "content": edited_loa})
        
        # Update the current LOA
        self.current_loa = edited_loa
        
        return edited_loa
    
    def save_loa(self, filename: str) -> None:
        """
        Saves the current LOA to a text file.
        
        Args:
            filename: Name of the file to save the LOA to
        """
        if not self.current_loa:
            print("No LOA has been generated yet. Please generate an LOA first.")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.current_loa)
            
        print(f"LOA saved to {filename}")
    
    def export_to_json(self, filename: str) -> None:
        """
        Exports the conversation history to a JSON file.
        
        Args:
            filename: Name of the file to save the conversation to
        """
        if not self.conversation_history:
            print("No conversation history available.")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2)
            
        print(f"Conversation history exported to {filename}")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------

def main():
    # Create a new LOA generator
    generator = LOAGenerator()
    
    # Example parameters for a digital hoarding LOA
    params = {
        "reference_number": "RE/DIGITAL_HOARDING/LOA/2024/001",
        "date": datetime.date.today(),
        "company_name": "Wallop Advertising Private Limited",
        "address_line1": "Office No. 02",
        "address_line2": "10th Floor, Pinnacle Corporate Park",
        "address_line3": "Near Trade Center, BKC, Bandra (E)",
        "city": "Mumbai",
        "pincode": "400051",
        "contact_person": "Mr. Arshad Khan",
        "contact_email": "arshad.khan@wallopadvertising.com",
        "contact_phone": "9029006666",
        "scenario": "Digital Hoarding",
        "scenario_description": "Providing Digital Hoardings at Retail Outlet",
        "location": "FC Road, Pune City",
        "size": "20' X 20' = 400 Sq. Ft.",
        "duration": "5",
        "payment_type": "Quarterly",
        "payment_amount": "310",
        "payment_unit": "Sq Ft/Per Year",
        "annual_increase": "5",
        "signatory_name": "Territory Manager",
        "signatory_position": "Territory Manager (Retail)",
        "organization": "Example Corporation Ltd."
    }
    
    # Generate the LOA
    loa = generator.generate_loa(params)
    print("--- Generated LOA ---")
    print(loa)
    
    # Example edit request
    edit_request = "Change the payment amount to 350 per square foot and add a clause about the vendor being responsible for any damage to the property."
    
    # Edit the LOA
    edited_loa = generator.edit_loa(edit_request)
    print("\n--- Edited LOA ---")
    print(edited_loa)
    
    # Save the LOA
    generator.save_loa("sample_loa.txt")
    
    # Export conversation history
    generator.export_to_json("conversation_history.json")


if __name__ == "__main__":
    main()


# -------------------------------------------------------------------------
# Streamlit App (for future implementation)
# -------------------------------------------------------------------------

"""
import streamlit as st
import datetime

def streamlit_app():
    st.title("LOA Generator for Outdoor Advertising")
    
    # Initialize LOA generator
    if 'loa_generator' not in st.session_state:
        st.session_state.loa_generator = LOAGenerator()
    
    # Sidebar for LOA generation
    with st.sidebar:
        st.header("LOA Parameters")
        
        # Reference and Date
        st.subheader("Reference Information")
        reference_number = st.text_input("Reference Number", value="LOA/2024/001")
        date = st.date_input("Date", value=datetime.date.today())
        
        # Recipient Information
        st.subheader("Recipient Information")
        company_name = st.text_input("Company Name", value="Wallop Advertising Pvt Ltd")
        address_line1 = st.text_input("Address Line 1", value="Office No. 02")
        address_line2 = st.text_input("Address Line 2", value="10th Floor, Pinnacle Corporate Park")
        address_line3 = st.text_input("Address Line 3", value="Near Trade Center, BKC, Bandra (E)")
        city = st.text_input("City", value="Mumbai")
        pincode = st.text_input("Pincode", value="400051")
        
        # Contact Person
        st.subheader("Contact Person")
        contact_person = st.text_input("Contact Person", value="Mr. Arshad Khan")
        contact_email = st.text_input("Contact Email", value="example@example.com")
        contact_phone = st.text_input("Contact Phone", value="9029006666")
        
        # Scenario Details
        st.subheader("Scenario Details")
        scenario_options = [
            "Digital Hoarding", 
            "Metro Barricade", 
            "Billboard", 
            "Retail Outlet Signage",
            "Custom Scenario"
        ]
        scenario = st.selectbox("Scenario", options=scenario_options)
        
        # Show custom scenario input if selected
        if scenario == "Custom Scenario":
            scenario_description = st.text_input("Describe Scenario")
        else:
            scenario_description = scenario
        
        location = st.text_input("Location", value="FC Road, Pune")
        size = st.text_input("Size", value="20' X 20' = 400 Sq. Ft.")
        duration = st.number_input("Duration (years)", min_value=1, max_value=20, value=5)
        
        # Payment Details
        st.subheader("Payment Details")
        payment_type_options = ["Annual", "Quarterly", "Monthly", "One-time"]
        payment_type = st.selectbox("Payment Type", options=payment_type_options)
        payment_amount = st.number_input("Payment Amount", min_value=0.0, value=310.0)
        payment_unit = st.text_input("Payment Unit", value="Sq Ft/Per Year")
        annual_increase = st.number_input("Annual Increase (%)", min_value=0.0, max_value=100.0, value=5.0)
        
        # Signatory Information
        st.subheader("Signatory Information")
        signatory_name = st.text_input("Signatory Name", value="Territory Manager")
        signatory_position = st.text_input("Signatory Position", value="Territory Manager (Retail)")
        organization = st.text_input("Organization", value="Example Corporation Ltd.")
        
        # Additional Terms
        st.subheader("Additional Terms")
        additional_terms = st.text_area("Additional Terms and Conditions")
        
        # Generate button
        generate_button = st.button("Generate LOA")
    
    # Main area for displaying and editing LOA
    if 'current_loa' not in st.session_state:
        st.session_state.current_loa = None
    
    # Generate LOA when button is clicked
    if generate_button:
        # Collect all parameters
        params = {
            "reference_number": reference_number,
            "date": date,
            "company_name": company_name,
            "address_line1": address_line1,
            "address_line2": address_line2,
            "address_line3": address_line3,
            "city": city,
            "pincode": pincode,
            "contact_person": contact_person,
            "contact_email": contact_email,
            "contact_phone": contact_phone,
            "scenario": scenario,
            "scenario_description": scenario_description,
            "location": location,
            "size": size,
            "duration": str(duration),
            "payment_type": payment_type,
            "payment_amount": str(payment_amount),
            "payment_unit": payment_unit,
            "annual_increase": str(annual_increase),
            "additional_terms": additional_terms,
            "signatory_name": signatory_name,
            "signatory_position": signatory_position,
            "organization": organization
        }
        
        # Generate LOA
        loa = st.session_state.loa_generator.generate_loa(params)
        st.session_state.current_loa = loa
    
    # Display current LOA if available
    if st.session_state.current_loa:
        st.header("Generated LOA")
        st.text_area("LOA Content", value=st.session_state.current_loa, height=500)
        
        # Edit functionality
        st.header("Edit LOA")
        edit_request = st.text_area("Describe the changes you want to make", height=100)
        edit_button = st.button("Apply Edit")
        
        if edit_button and edit_request:
            edited_loa = st.session_state.loa_generator.edit_loa(edit_request)
            st.session_state.current_loa = edited_loa
            st.experimental_rerun()
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download as Text"):
                st.download_button(
                    label="Download LOA",
                    data=st.session_state.current_loa,
                    file_name="generated_loa.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("Export Conversation"):
                # First save to a temporary file
                st.session_state.loa_generator.export_to_json("temp_conversation.json")
                # Then read and provide for download
                with open("temp_conversation.json", "r") as f:
                    conversation_json = f.read()
                
                st.download_button(
                    label="Download Conversation",
                    data=conversation_json,
                    file_name="conversation_history.json",
                    mime="application/json"
                )
    else:
        st.info("Configure parameters and click 'Generate LOA' to create a new Letter of Authorization.")


if __name__ == "__main__":
    streamlit_app()
"""