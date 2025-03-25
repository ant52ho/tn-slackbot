import streamlit as st
import pandas as pd
import snowflake.connector
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os
import re
from datasets import Dataset

# Page configuration
st.set_page_config(
    page_title="TextNow Topic Classifier",
    page_icon="🔍",
    layout="wide"
)

# Current date/time and user as hardcoded
current_datetime = "2025-03-25 02:02:13"  # Hardcoded as requested
current_user = "harishaaram"  # Hardcoded as requested

# Initialize the session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'row_count' not in st.session_state:
    st.session_state.row_count = {}
if 'id2label' not in st.session_state:
    st.session_state.id2label = {}
if 'label2id' not in st.session_state:
    st.session_state.label2id = {}

# Channel mapping
channel_mapping = {
    "C7X5D4ZUY": "#ask-data-team",
    "C4WTA9ZBJ": "#fraud-squad",
    "CFBF2NKRS": "#ask-sre",
    "C6FBB5ZGW": "#ask-backend",
    "C0196L2UC3G": "#calling-team",
    "C02CB63HSNL": "#ask-messaging"
}

# Reverse mapping for display
reverse_channel_mapping = {v: k for k, v in channel_mapping.items()}

# Hardcoded Snowflake connection details
snowflake_config = {
    "account": "zja64434.us-east-1",
    "user": "tn_dataloader",
    "password": "YMzJ5Tnjs2ShvjvP",  # Replace with actual password in real app
    "warehouse": "DEV_WH_SMALL",
    "database": "DEV",
    "schema": "PUBLIC"
}

# Function to connect to Snowflake
def connect_to_snowflake():
    try:
        conn = snowflake.connector.connect(
            user=snowflake_config["user"],
            password=snowflake_config["password"],
            account=snowflake_config["account"],
            warehouse=snowflake_config["warehouse"],
            database=snowflake_config["database"],
            schema=snowflake_config["schema"]
        )
        st.session_state.conn = conn
        st.session_state.connected = True
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {str(e)}")
        return None

# Function to get row counts per channel
def get_channel_row_counts():
    if not st.session_state.connected:
        conn = connect_to_snowflake()
        if not conn:
            return None
    else:
        conn = st.session_state.conn
    
    try:
        cur = conn.cursor()
        query = """
        SELECT CHANNELID, COUNT(*) as row_count
        FROM DEV.PUBLIC.CUSTOM_SLACK_MESSAGE
        GROUP BY CHANNELID
        ORDER BY row_count DESC
        """
        cur.execute(query)
        results = cur.fetchall()
        
        # Convert to dictionary
        row_counts = {}
        for channel, count in results:
            topic = channel_mapping[channel]
            row_counts[topic] = count
        
        return row_counts
    except Exception as e:
        st.error(f"Error getting row counts: {str(e)}")
        return None

# App header
st.title("TextNow Topic Classifier")
st.markdown(f"**System Date (UTC):** {current_datetime}")
st.markdown(f"**User:** {current_user}")

# Sidebar with information
with st.sidebar:
    st.header("Channel Information")
    st.markdown("""
    This classifier maps user queries to these TextNow Slack channels:
    - **#ask-data-team**: Data Analysis and Insights  
    - **#fraud-squad**: Fraud Detection and Prevention
    - **#ask-sre**: Site Reliability and Infrastructure
    - **#ask-backend**: Backend Systems and Development
    - **#calling-team**: Calling related Developments
    - **#ask-messaging**: Messaging related developments
    """)
    
    # Add clear history button
    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History cleared!")

# Main content
tab1, tab2 = st.tabs(["Classifier", "Data Management"])

# Data Management Tab
with tab2:
    st.header("Snowflake Data Management")
    
    # Button to get row counts
    if st.button("Get Message Count Per Channel"):
        with st.spinner("Connecting to Snowflake and counting messages..."):
            row_counts = get_channel_row_counts()
            
            if row_counts:
                st.session_state.row_count = row_counts
                
                # Display total count
                total_count = sum(row_counts.values())
                st.metric("Total Slack Messages", f"{total_count:,}")
                
                # Create DataFrame for display
                counts_df = pd.DataFrame({
                    'Channel': list(row_counts.keys()),
                    'Message Count': list(row_counts.values())
                })
                
                # Display as chart
                st.subheader("Message Count by Channel")
                st.bar_chart(counts_df.set_index('Channel'))
                
                # Display as table
                counts_df = counts_df.sort_values('Message Count', ascending=False)
                st.table(counts_df)
    
    # Display previously fetched counts if available
    elif st.session_state.row_count:
        # Display total count
        total_count = sum(st.session_state.row_count.values())
        st.metric("Total Slack Messages", f"{total_count:,}")
        
        # Create DataFrame for display
        counts_df = pd.DataFrame({
            'Channel': list(st.session_state.row_count.keys()),
            'Message Count': list(st.session_state.row_count.values())
        })
        
        # Display as chart
        st.subheader("Message Count by Channel")
        st.bar_chart(counts_df.set_index('Channel'))
        
        # Display as table
        counts_df = counts_df.sort_values('Message Count', ascending=False)
        st.table(counts_df)
        
    # Load data for training
    st.header("Model Training")
    
    if st.button("Load Data and Train Transformer Model"):
        with st.spinner("Connecting to Snowflake, loading data, and training model..."):
            try:
                # Connect to Snowflake if not already connected
                if not st.session_state.connected:
                    conn = connect_to_snowflake()
                    if not conn:
                        st.error("Cannot proceed without Snowflake connection")
                        st.stop()
                else:
                    conn = st.session_state.conn
                
                # Create a cursor object
                cur = conn.cursor()
                
                # Execute the query to fetch Slack messages
                query = """
                SELECT MSG, CHANNELID
                FROM DEV.PUBLIC.CUSTOM_SLACK_MESSAGE_FORMATTED
                WHERE THREADID IS NULL AND MSG IS NOT NULL
                """
                cur.execute(query)
                
                # Fetch the data
                data = cur.fetchall()
                
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=['message', 'channel_id'])
                
                # Clean the data
                df = df.dropna()
                
                # Map channel IDs to topic names
                df['topic'] = df['channel_id'].map(channel_mapping)
                
                # Create label mappings for the model
                unique_topics = df['topic'].unique()
                id2label = {i: label for i, label in enumerate(unique_topics)}
                label2id = {label: i for i, label in id2label.items()}
                
                # Add numeric labels
                df['label'] = df['topic'].map(label2id)
                
                # Store mappings in session state
                st.session_state.id2label = id2label
                st.session_state.label2id = label2id
                
                # Store in session state
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                # Display the data
                st.write(f"Loaded {len(df)} messages from Snowflake")
                st.dataframe(df[['message', 'channel_id', 'topic', 'label']].head(10))
                
                # Convert to Hugging Face dataset
                dataset = Dataset.from_pandas(df)
                
                # Split into train and test
                train_test = dataset.train_test_split(test_size=0.2)
                
                # Load pre-trained model and tokenizer
                model_name = "distilbert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Tokenize the data
                def tokenize_function(examples):
                    return tokenizer(examples["message"], padding="max_length", truncation=True)
                
                tokenized_datasets = train_test.map(tokenize_function, batched=True)
                
                # Configure model for fine-tuning
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=len(unique_topics),
                    id2label=id2label,
                    label2id=label2id
                )
                
                # Fine-tune model (placeholder - for a real implementation this would use proper training)
                # This is just for demo purposes - in a real scenario, you'd use Trainer from transformers
                st.info("For demo purposes, we're setting up a pre-trained model without full fine-tuning")
                
                # Create a classification pipeline
                classifier = pipeline(
                    "text-classification", 
                    model=model, 
                    tokenizer=tokenizer
                )
                
                # Store in session state
                st.session_state.model = classifier
                st.session_state.tokenizer = tokenizer
                st.session_state.training_complete = True
                
                st.success("Model prepared successfully!")
                
            except Exception as e:
                st.error(f"Error processing data and setting up model: {str(e)}")
    
    # Use demo data for hackathon
    if st.button("Use Demo Data with Pre-trained Transformer (for Hackathon)"):
        with st.spinner("Setting up demo data and model..."):
            try:
                # Create synthetic training data for demo purposes
                demo_data = []
                
                # Data team
                data_messages = [
                    "Need help analyzing user engagement metrics",
                    "Can someone help me understand this data anomaly?",
                    "How do we interpret the latest retention numbers?",
                    "Looking for insights on conversion rates",
                    "Need to build a dashboard for monthly active users",
                    "Can we predict churn based on these variables?",
                    "How should we segment our user base?",
                    "Need help with A/B test statistical significance",
                    "What's the best visualization for this funnel data?",
                    "Can someone run a query on the latest user behaviors?"
                ]
                
                # Fraud squad
                fraud_messages = [
                    "Seeing suspicious sign-up patterns from these IPs",
                    "Possible account takeover attempts detected",
                    "How do we handle this potential fraud case?",
                    "New fraud pattern emerging in account creation",
                    "Need to update our fraud detection rules",
                    "Seeing unusual number of messages and calls",
                    "Possible SIM swapping attacks being reported",
                    "Unusual activity on these accounts needs review",
                    "How do we verify the legitimacy of these users?",
                    "Need to implement additional security measures"
                ]
                
                # SRE
                sre_messages = [
                    "Server latency has increased in the last hour",
                    "Need to scale up the messaging service",
                    "Database performance issue affecting calls",
                    "Investigating the recent outage root cause",
                    "Load balancer configuration needs updating",
                    "Monitoring alert for high CPU usage",
                    "Need to implement failover for this service",
                    "How do we improve our incident response?",
                    "Working on improving system resilience",
                    "Need to update our SLO definitions"
                ]
                
                # Backend
                backend_messages = [
                    "API rate limiting issue in the messaging service",
                    "How do we optimize this database query?",
                    "Need to refactor this authentication flow",
                    "Working on the new messaging microservice",
                    "How should we implement this new feature?",
                    "Database migration plan needs review",
                    "Cache invalidation strategy for user profiles",
                    "Backend performance bottleneck in the call service",
                    "How do we handle concurrent updates to user status?",
                    "Need to implement webhook notifications"
                ]

                # Messaging team
                messaging_examples = [
                    "Issue with message delivery status updates",
                    "How do we improve message sync across devices?",
                    "Need to implement rich media previews in messages",
                    "Working on message retention policy implementation",
                    "Bug in message threading functionality",
                    "How do we handle message encryption end-to-end?",
                    "Performance issues with high-volume messaging users",
                    "Need to implement read receipts for group messages",
                    "Working on emoji reaction functionality in messaging",
                    "How do we handle message delivery in poor network conditions?"
                ]

                # Calling team
                calling_examples = [
                    "Voice quality issues in international calls",
                    "How do we reduce call setup latency?",
                    "Investigating dropped calls in the Android app",
                    "Need to implement call forwarding feature",
                    "Working on conference call stability",
                    "How do we handle call audio during network switching?",
                    "Echo cancellation not working properly in VoIP calls",
                    "Need to optimize bandwidth usage during video calls",
                    "Implementation plan for call recording feature",
                    "How do we improve caller ID accuracy?"
                ]

                message_sets = [
                    (data_messages, "Data Analysis and Insights", "#ask-data-team"),
                    (fraud_messages, "Fraud Detection and Prevention", "#fraud-squad"),
                    (sre_messages, "Site Reliability and Infrastructure", "#sre"),
                    (backend_messages, "Backend Systems and Development", "#ask-backend"),
                    (messaging_examples, "Messaging related developments", "#ask-messaging"),
                    (calling_examples, "Calling related Developments", "#calling-team"),
                ]
                
                # Create records
                for messages, topic, channel in message_sets:
                    for msg in messages:
                        timestamp = datetime.datetime.now() - datetime.timedelta(days=30) + datetime.timedelta(days=30*len(demo_data)/70)
                        demo_data.append({
                            'message': msg,
                            'channel_id': channel,
                            'topic': topic
                        })
                
                # Convert to DataFrame
                demo_df = pd.DataFrame(demo_data)
                
                # Create label mappings for the model
                unique_topics = demo_df['topic'].unique()
                id2label = {i: label for i, label in enumerate(unique_topics)}
                label2id = {label: i for i, label in id2label.items()}
                
                # Add numeric labels
                demo_df['label'] = demo_df['topic'].map(label2id)
                
                # Store mappings in session state
                st.session_state.id2label = id2label
                st.session_state.label2id = label2id
                
                # Store in session state
                st.session_state.df = demo_df
                st.session_state.data_loaded = True
                
                # Display the data
                st.write(f"Created {len(demo_df)} demo messages")
                st.dataframe(demo_df[['message', 'channel_id', 'topic', 'label']].head(10))
                
                # For demo purposes, we'll use a zero-shot classification pipeline
                # This avoids the need to actually train the model, which can be time-consuming
                classifier = pipeline("zero-shot-classification")
                
                # Store in session state
                st.session_state.model = classifier
                st.session_state.training_complete = True
                
                st.success("Demo model set up successfully!")
                
                # Show data distribution
                st.subheader("Message Distribution by Channel")
                channel_counts = demo_df['channel_id'].value_counts().reset_index()
                channel_counts.columns = ['Channel', 'Count']
                st.bar_chart(channel_counts.set_index('Channel'))
                
            except Exception as e:
                st.error(f"Error setting up demo model: {str(e)}")

# Classifier Tab
with tab1:
    st.header("Message Topic Classification")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Text input
        user_input = st.text_area(
            "Enter your message here:",
            height=150,
            placeholder="Example: My iOS app keeps crashing when I try to make a call. The app version is 5.7.2 on iPhone 12."
        )
        
        # When model isn't trained yet, provide a message
        if not st.session_state.training_complete:
            st.info("Please go to the 'Data Management' tab to set up the model first, or use the demo option.")
        
        # Classification button
        classify_button = st.button("Classify Message")
        if classify_button and user_input and st.session_state.training_complete:
            with st.spinner("Analyzing message..."):
                # Get the model
                classifier = st.session_state.model
                
                # Check if we're using zero-shot (demo) or fine-tuned model
                if classifier.task == "zero-shot-classification":
                    # For zero-shot pipeline
                    topics = list(channel_mapping.values())
                    result = classifier(user_input, topics)
                    
                    topic = result["labels"][0]
                    confidence = result["scores"][0]
                    channel = reverse_channel_mapping[topic]
                    
                    # Create results for history
                    all_topics = result["labels"]
                    all_confidences = result["scores"]
                else:
                    # For fine-tuned text classification model
                    result = classifier(user_input)
                    st.write(result)
                    # Check if the result label is already a topic name or needs parsing
                    if result[0]['label'] in st.session_state.label2id:
                        # Label is already a topic name
                        topic = result[0]['label']
                        label_id = st.session_state.label2id[topic]
                    else:
                        # Try to parse label_id from the result
                        try:
                            label_id = int(result[0]['label'].split('_')[-1])
                            topic = st.session_state.id2label[label_id]
                        except (ValueError, KeyError):
                            # If parsing fails, use the label as topic directly
                            topic = result[0]['label']
                            label_id = 0  # Default ID
                    
                    confidence = result[0]['score']
                    channel = reverse_channel_mapping.get(topic, "unknown-channel")
                    
                    # We don't have confidences for all classes in this simplified demo
                    # In a real implementation, you'd use model.predict_proba
                    all_topics = list(st.session_state.id2label.values())
                    all_confidences = [0.1] * len(all_topics)  # Placeholder
                    all_confidences[label_id] = confidence
                
                # Add to history
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                history_item = {
                    "time": timestamp,
                    "message": user_input[:50] + "..." if len(user_input) > 50 else user_input,
                    "topic": topic,
                    "channel": channel,
                    "confidence": confidence,
                    "full_message": user_input,
                    "all_topics": all_topics,
                    "all_confidences": all_confidences
                }
                st.session_state.history.insert(0, history_item)  # Add to beginning
            
            st.success("Message classified successfully!")
    
    with col2:
        st.header("Classification Results")
        
        if st.session_state.history:
            latest = st.session_state.history[0]
            
            # Display latest results in a card-like format
            st.subheader("Latest Classification")
            st.markdown(f"""
            <div style="border:1px solid #ddd; padding:15px; border-radius:5px;">
                <h4 style="margin-top:0;">Recommended Channel: <span style="color:#ff4b4b;">{latest['channel']}</span></h4>
                <p><strong>Topic:</strong> {latest['topic']}<br/>
                <strong>Confidence:</strong> {latest['confidence']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence for all channels
            st.subheader("All Channels")
            
            # Create a DataFrame for display
            results_data = []
            for topic, confidence in zip(latest['all_topics'], latest['all_confidences']):
                if isinstance(topic, str) and topic in reverse_channel_mapping:
                    channel = reverse_channel_mapping[topic]
                    results_data.append({
                        'Channel': channel,
                        'Topic': topic,
                        'Confidence': confidence
                    })
            
            results_df = pd.DataFrame(results_data)
            
            # Sort by confidence
            results_df = results_df.sort_values('Confidence', ascending=False)
            
            # Display as bar chart
            st.bar_chart(results_df.set_index('Channel')['Confidence'])
            
            # Display as table
            results_df['Confidence'] = results_df['Confidence'].apply(lambda x: f"{x:.2%}")
            st.table(results_df)
        else:
            st.info("No classifications yet. Enter a message and click 'Classify Message'.")
    
    # Display history
    st.header("Classification History")
    if st.session_state.history:
        history_table = ""
        for i, item in enumerate(st.session_state.history):
            if i < 5:  # Limit to 5 most recent items
                history_table += f"""
                <tr>
                    <td>{item['time']}</td>
                    <td>{item['message']}</td>
                    <td>{item['channel']}</td>
                    <td>{item['confidence']:.2%}</td>
                </tr>
                """
        
        st.markdown(f"""
        <table style="width:100%">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Message</th>
                    <th>Channel</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                {history_table}
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.history) > 5:
            st.caption(f"Showing 5 most recent of {len(st.session_state.history)} classifications.")
    else:
        st.info("No classification history yet.")
    
    # # Add example messages
    # st.header("Example Messages")
    # example_messages = {
    #     "Data Question": "Could someone help me understand the recent drop in active users? I need to analyze the trend.",
    #     "Fraud Alert": "We're seeing an unusual pattern of sign-ups from the same IP range. Possible account farming?",
    #     "Outage Report": "The messaging service seems to be down. Multiple users reporting delivery failures.",
    #     "Backend Question": "How do I modify the rate limiting for the kafka streams? The current settings are too restrictive.",
    # }
    #
    # st.markdown("Click an example to use it:")
    # cols = st.columns(3)
    # for i, (title, message) in enumerate(example_messages.items()):
    #     col = cols[i % 3]
    #     if col.button(title):
    #         st.session_state.example_selected = message
    #         st.experimental_rerun()
    #
    # # Use the selected example if there is one
    # if 'example_selected' in st.session_state:
    #     st.session_state.pop('example_selected')