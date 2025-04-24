import os
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

# First, make sure you have your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create a simple Pydantic model for testing
class TicketDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    # ticket fields
    title: str = Field(description="title of the jira ticket")
    description: str = Field(description="description of the jira ticket")
    priority: str = Field(description="priority of the jira ticket, from P0 to P4")

    # ticket creation flags
    skip_details: bool = Field(description="User inputted flag to skip inputting details for this ticket")
    fraud_noc_pinged: bool = Field(description="User inputted flag to ping fraud_noc")
    
    # follow up question for model to ask
    follow_up_question: Optional[str] = Field(description="follow up question to ask the user to provide more details")


def main():
    # Initialize the components
    llm = ChatOpenAI(
        model="gpt-4-0125-preview",
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    
    parser = PydanticOutputParser(pydantic_object=TicketDetails)

    messages = "I am detecting high levels of fraud on my account"
    # Create a simple prompt
    # fill out jira ticket responses based on description
    prompt_template = """
            Based on the context provided below, please extract and refine the key details needed 
            to create your Jira ticket. Determine if the issue is on an individual scope (e.g., 'I can't log in', 'I am getting rate limited') 
            or a broader fraud issue (e.g., a spike in registrations affecting a company metric).
            For individual issues: If data like your account details or device information is missing, include a follow-up question: 
            'Could you please provide any account details and the device information?' Otherwise, leave it blank.
            For broader issues: If details such as which dashboard you're using or the specific date range of concern are missing, 
            include a follow-up question: 
            'Could you please specify which dashboard you are using and the specific date range of concern?' Otherwise, leave it blank.
            Use second-person language and only ask for details not already provided. Provide a concise title and a descriptive summary. 
            Format your response as JSON with these fields:
            - title (string): Summary of the issue.
            - description (string): Detailed and refined problem description.
            - priority (string): Priority of the issue, from P0 to P4.
            - follow_up_question (string, optional): Missing information query. Leave blank if none.
            - skip_details (boolean): True if @fraud_noc mentioned, False otherwise (default: False).
            - fraud_noc_pinged (boolean): True if @fraud_noc mentioned, False otherwise (default: False).
            Use second-person language and only request information not already provided.
            
            Context:
            {messages}

            Format instructions
            {format_instructions}
        """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["messages"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Test the chain
    print("Creating chain...")
    chain = prompt | llm | parser
    
    print("Invoking chain...")
    result = chain.invoke({"messages": messages})
    
    print("\nResult type:", type(result))
    print("Result content:", result)
    print("\nAccessing fields:")
    print(f"Title: {result.title}")
    print(f"Description: {result.description}")
    print(f"Priority: {result.priority}")
    print(f"Follow up question: {result.follow_up_question}")
    print(f"Skip details: {result.skip_details}")
    print(f"Fraud Noc pinged: {result.fraud_noc_pinged}")

if __name__ == "__main__":
    main() 