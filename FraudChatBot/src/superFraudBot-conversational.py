import streamlit as st
import json
import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from jira import JIRA
from rag_helpers import setEnvs  # load env vars from your file

# --- Conversational Chatbot to Handle Jira Tickets and Escalations ---

class JiraTicketBot:
    def __init__(self, llm, logger: logging.Logger):
        self.llm = llm
        self.logger = logger

    def dynamic_conversation(self, state: dict) -> dict:
        """
        Build context from the current state and let the LLM extract
        details for the Jira ticket.
        """
        context_parts = []
        if state.get("issue") is not None:
            context_parts.append("Issue: " + state["issue"])
        if state.get("account_info"):
            context_parts.append("Tester Account Info: " + state["account_info"])
        if state.get("device_info"):
            context_parts.append("Device/Environment Info: " + state["device_info"])
        if state.get("error_info"):
            context_parts.append("Error Info: " + state["error_info"])
        if state.get("priority"):
            context_parts.append("Priority: " + state["priority"])
        context = "\n".join(context_parts)

        prompt_template = ChatPromptTemplate.from_template(
            "Based on the context provided below, please extract and refine the key details needed "
            "to create your Jira ticket. Determine if the issue is on an individual scope (e.g., 'I can't log in', 'I am getting rate limited') "
            "or a broader fraud issue (e.g., a spike in registrations affecting a company metric).\n\n"
            "For individual issues: If data like your account details or device information is missing, include a follow-up question: "
            "'Could you please provide any account details and the device information?' Otherwise, leave it blank.\n\n"
            "For broader issues: If details such as which dashboard you're using or the specific date range of concern are missing, "
            "include a follow-up question: "
            "'Could you please specify which dashboard you are using and the specific date range of concern?' Otherwise, leave it blank.\n\n"
            "Use second-person language and only ask for details not already provided. Provide a concise title and a descriptive summary. "
            "Respond as JSON with keys 'title', 'description', and 'follow_up_question'.\n\n"
            "Context:\n{context}\n"
        )
        prompt = prompt_template.invoke({"context": context})
        result = self.llm.invoke(prompt)
        try:
            parsed = json.loads(result.content.strip())
        except Exception as e:
            self.logger.error("Failed to parse LLM response: %s", e)
            parsed = {
                "title": "Untitled Issue",
                "description": context,
                "follow_up_question": "Could you please provide more details about the issue?"
            }
        return parsed

    def confirm_ticket(self, state: dict) -> dict:
        ticket_payload = {
            "title": state.get("title", "Untitled Issue"),
            "description": state.get("description", ""),
            "priority": state.get("priority", "P4")
        }
        return ticket_payload

# --- Streamlit App Initialization ---

def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "inputs" not in st.session_state:
        st.session_state.inputs = {}
    if "done" not in st.session_state:
        st.session_state.done = False
    if "ticket_result" not in st.session_state:
        st.session_state.ticket_result = ""
    if "ticket_submitted" not in st.session_state:
        st.session_state.ticket_submitted = False
    if "ticket_bot" not in st.session_state:
        setEnvs(path="../rag.env")
        logger = logging.getLogger(__name__)
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        st.session_state.ticket_bot = JiraTicketBot(llm=llm, logger=logger)

def interpret_priority(input_str: str) -> str:
    mapping = {
        "critical": "P0",
        "urgent": "P1",
        "high": "P2",
        "medium": "P3",
        "low": "P4",
        "p0": "P0",
        "p1": "P1",
        "p2": "P2",
        "p3": "P3",
        "p4": "P4"
    }
    input_lower = input_str.strip().lower()
    for key, value in mapping.items():
        if key in input_lower:
            return value
    return "P4"

# --- Main Logic ---

def main():
    st.set_page_config(page_title="Support Bot", layout="centered")
    st.title("🛠️ Fraud Squad Support Chatbot")
    init_state()

    jira_server = os.environ.get("JIRA_SERVER")
    jira_username = os.environ.get("JIRA_USERNAME")
    jira_password = os.environ.get("JIRA_PASSWORD")
    jira_project = os.environ.get("JIRA_PROJECT")
    bot = st.session_state.ticket_bot
    inputs = st.session_state.inputs
    history = st.session_state.chat_history

    # Helper: Check for "@fraud_noc" in user input to begin escalation.
    def check_fraud_noc(prompt_text):
        trimmed = prompt_text.strip().lower()
        if trimmed == "@fraud_noc":
            inputs["fraud_noc_pinged"] = True
            return "exact"
        elif "@fraud_noc" in trimmed:
            inputs["fraud_noc_pinged"] = True
            return True
        return False

    # Display conversation history.
    for entry in history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["message"])

    # --- Step 1: Ask for description ---
    if not inputs.get("issue") and not inputs.get("skip_details"):
        question = (
            "Hi! It looks like we couldn’t find a similar issue. Could you please describe the problem you're facing? "
            "You can copy and paste your original message if you've already summarized it. "
            "At any point, please ping @fraud_noc to skip this process."
        )
        with st.chat_message("assistant"):
            st.markdown(question)
        prompt = st.chat_input("Your message...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            history.append({"role": "user", "message": prompt})
            result = check_fraud_noc(prompt)
            if result == "exact":
                # If user types exactly "@fraud_noc", clear any issue info and set skip_details flag.
                inputs["issue"] = ""
                inputs["skip_details"] = True
                inputs["draft_confirmed"] = True
                inputs["priority"] = "P4"  # default priority
            else:
                inputs["issue"] = prompt
            st.rerun()
        return

    # --- Step 2: Ask for priority (if not skipping details) ---
    if not inputs.get("priority") and not inputs.get("skip_details"):
        question = (
            "Could you please specify the priority of this issue? "
            "For example, you can say Critical (P0), Urgent (P1), High (P2), Medium (P3), or Low (P4). "
            "At any point, please ping @fraud_noc to skip this process."
        )
        with st.chat_message("assistant"):
            st.markdown(question)
        prompt = st.chat_input("Your response...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            history.append({"role": "user", "message": prompt})
            result = check_fraud_noc(prompt)
            if result == "exact":
                inputs["priority"] = "P4"
                inputs["skip_details"] = True
                inputs["draft_confirmed"] = True
            else:
                inputs["priority"] = interpret_priority(prompt)
            st.rerun()
        return

    # --- Step 3: Confirm or add details ---
    if not inputs.get("draft_confirmed") and not inputs.get("skip_details"):
        conv_result = bot.dynamic_conversation(inputs)
        inputs.update(conv_result)
        if conv_result.get("follow_up_question", "").strip() and not inputs.get("follow_up_answered"):
            with st.chat_message("assistant"):
                st.markdown(conv_result["follow_up_question"] + " (At any point, you may ping @fraud_noc to skip this process.)")
            prompt = st.chat_input("Your additional details...")
            if prompt:
                with st.chat_message("user"):
                    st.markdown(prompt)
                history.append({"role": "user", "message": prompt})
                result = check_fraud_noc(prompt)
                if result == "exact":
                    inputs["draft_confirmed"] = True
                    inputs["skip_details"] = True  # Set skip_details if the user exactly types "@fraud_noc"
                else:
                    inputs["issue"] += "\n" + prompt
                    inputs["follow_up_answered"] = True
                st.rerun()
            return
        else:
            ticket_data = bot.confirm_ticket(inputs)
            summary = (
                f"Here's what I drafted for your ticket:\n\n"
                f"**Title:** {ticket_data['title']}\n\n"
                f"**Description:** {ticket_data['description']}\n\n"
                f"**Priority:** {ticket_data.get('priority', 'P4')}\n\n"
                "Does this look correct? Please reply with 'yes' to confirm, or add additional details to improve the ticket. "
                "At any point, you may ping @fraud_noc to skip this process."
            )
            with st.chat_message("assistant"):
                st.markdown(summary)
            prompt = st.chat_input("Your response...")
            if prompt:
                with st.chat_message("user"):
                    st.markdown(prompt)
                history.append({"role": "user", "message": prompt})
                result = check_fraud_noc(prompt)
                if result == "exact" or "yes" in prompt.lower():
                    inputs["draft_confirmed"] = True
                    if result == "exact":
                        inputs["skip_details"] = True
                else:
                    inputs["issue"] += "\n" + prompt
                st.rerun()
            return

    # --- Step 4: Create the Jira ticket (only once) ---
    if not st.session_state.ticket_submitted:
        with st.chat_message("assistant"):
            with st.spinner("Creating your ticket..."):
                conv_result = bot.dynamic_conversation(inputs)
                inputs.update(conv_result)
                ticket_data = bot.confirm_ticket(inputs)
                # If skip_details flag is set AND no significant issue text was collected, override ticket data with defaults.
                if inputs.get("skip_details") and (not inputs.get("issue") or len(inputs.get("issue").strip()) < 10):
                    ticket_data["title"] = "Fraud Squad - Low Context Please See Channel"
                    ticket_data["description"] = "Slack thread: https://slack.com/fake_link"
                try:
                    jira_priority = ticket_data.get("priority", "P4").upper()
                    description = ticket_data.get("description", "")
                    extra_note = ""
                    if inputs.get("fraud_noc_pinged"):
                        extra_note = ("\n\nNOTE: @fraud_noc was pinged. The issue will be reviewed in the channel. "
                                      "Please continue the conversation with the team in the thread!")
                    elif jira_priority in ["P0", "P1", "P2"]:
                        extra_note = f"\n\nNOTE: Priority {jira_priority} detected. @fraud_noc has been autopinged."
                    description += extra_note
                    jira = JIRA(server=jira_server, basic_auth=(jira_username, jira_password))
                    new_issue = jira.create_issue(
                        project=jira_project,
                        summary=ticket_data["title"],
                        description=description,
                        issuetype={"name": "Task"},
                        priority={"name": jira_priority}
                    )
                    if inputs.get("fraud_noc_pinged"):
                        message = (f"✅ Ticket `{new_issue.key}` created. @fraud_noc was pinged. "
                                   "Please continue the conversation with the team in the thread!")
                    elif jira_priority in ["P0", "P1", "P2"]:
                        message = (f"✅ Ticket `{new_issue.key}` created successfully! @fraud_noc has been autopinged due to high priority.")
                    else:
                        message = f"✅ Ticket `{new_issue.key}` created successfully!"
                except Exception as e:
                    message = f"❌ Error creating ticket: {e}"
                st.session_state.ticket_result = message
                st.session_state.done = True
                st.session_state.ticket_submitted = True
                st.markdown(message)


    # --- Step 5: Final message and start over option ---
    if st.session_state.done:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.ticket_result)
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
