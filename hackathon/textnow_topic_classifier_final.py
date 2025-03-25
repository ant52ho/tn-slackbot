import streamlit as st
import pandas as pd
import snowflake.connector
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import re

# Page configuration
st.set_page_config(
    page_title="TextNow Slack-Channel Classifier",
    page_icon="🔍",
    layout="wide"
)

# Current date/time and user as specified
current_datetime = "2025-03-24 19:22:48"  # Specified datetime
current_user = "harishaaram"  # Specified username

# Initialize the session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'row_count' not in st.session_state:
    st.session_state.row_count = None

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

# App header
st.title("TextNow Slack-Channel Classifier")
st.markdown(f"**System Date (UTC):** {current_datetime}")
# st.markdown(f"**User:** {current_user}")

# Hardcoded Snowflake connection parameters (these will be used internally)
snowflake_config = {
    "account": "zja64434.us-east-1",
    "user": "tn_dataloader",
    "password": "YMzJ5Tnjs2ShvjvP",  # Replace with actual password in real app
    "warehouse": "DEV_WH_SMALL",
    "database": "DEV",
    "schema": "PUBLIC"
}

# Sidebar with information
with st.sidebar:
    st.header("Snowflake Connection")
    st.markdown("Connection parameters are configured securely in the application.")

    # Connect button with hardcoded credentials
    if st.button("Check Snowflake Connection"):
        try:
            with st.spinner("Connecting to Snowflake..."):
                # Create Snowflake connection with hardcoded parameters
                conn = snowflake.connector.connect(
                    user=snowflake_config["user"],
                    password=snowflake_config["password"],
                    account=snowflake_config["account"],
                    warehouse=snowflake_config["warehouse"],
                    database=snowflake_config["database"],
                    schema=snowflake_config["schema"]
                )

                # Save connection in session state
                st.session_state.conn = conn
                st.session_state.connected = True

                # Get row count from table
                cur = conn.cursor()
                query = """
                SELECT channel_name, COUNT(*) as message_count
                FROM DEV.PUBLIC.CUSTOM_SLACK_MESSAGE JOIN slack_channel_mapping USING (CHANNELID)
                GROUP BY channel_name
                ORDER BY message_count DESC
                """
                cur.execute(query)
                results = cur.fetchall()

            df = pd.DataFrame(results, columns=['channel_name', 'message_count'])
            st.success("Connected to Snowflake successfully!")
            st.dataframe(df)

        except Exception as e:
            st.error(f"Failed to connect to Snowflake: {str(e)}")

    st.markdown("---")

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

    # Row count display
    if st.session_state.row_count is not None:
        st.success(f"Connected to Snowflake database")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Total Slack Messages in Database", f"{st.session_state.row_count:,}")
    else:
        st.info("Click 'Check Snowflake Connection' in the sidebar to see the total message count")

    if not st.session_state.connected:
        st.warning("Please check Snowflake connection in the sidebar first")
    else:
        # Fetch and display data
        if st.button("Load Slack Messages for Training"):
            with st.spinner("Loading data from Snowflake..."):
                try:
                    # Create a cursor object
                    cur = st.session_state.conn.cursor()

                    # Execute the query to fetch Slack messages
                    query = """
                    SELECT MSG, CHANNELID, TIMESTAMP
                    FROM DEV.PUBLIC.CUSTOM_SLACK_MESSAGE_FORMATTED
                    """
                    cur.execute(query)

                    # Fetch the data
                    data = cur.fetchall()

                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=['message', 'channel_id', 'timestamp'])

                    # Clean the data
                    df = df.dropna()

                    # Add topic column based on channel mapping
                    df['topic'] = df['channel_id'].map(channel_mapping)

                    # Store in session state
                    st.session_state.df = df
                    st.session_state.data_loaded = True

                    # Display the data
                    st.write(f"Loaded {len(df)} messages from Snowflake")
                    st.dataframe(df[['message', 'channel_id', 'topic']].head(10))

                    # Show data distribution
                    st.subheader("Message Distribution by Channel")
                    channel_counts = df['topic'].value_counts().reset_index()
                    channel_counts.columns = ['Channel', 'Count']
                    st.bar_chart(channel_counts.set_index('Channel')['Count'])
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")

        # Train model once data is loaded
        if st.session_state.data_loaded:
            if st.button("Train Classification Model"):
                with st.spinner("Training model..."):
                    try:
                        # Get the data
                        # Execute the query to fetch Slack messages
                        cur = st.session_state.conn.cursor()
                        query = """
                        SELECT MSG, CHANNELID, TIMESTAMP
                        FROM DEV.PUBLIC.CUSTOM_SLACK_MESSAGE_FORMATTED
                        WHERE THREADID IS NULL AND MSG IS NOT NULL
                        """
                        cur.execute(query)

                        # Fetch the data
                        data = cur.fetchall()

                        # Convert to DataFrame
                        df_train = pd.DataFrame(data, columns=['message', 'channel_id', 'timestamp'])
                        df_train['topic'] = df_train['channel_id'].map(channel_mapping)

                        # Split into train and test sets
                        X_train, X_test, y_train, y_test = train_test_split(
                            df_train['message'], df_train['topic'], test_size=0.2, random_state=42
                        )

                        # Create pipeline with TF-IDF and Naive Bayes
                        model = Pipeline([
                            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                            ('classifier', MultinomialNB())
                        ])

                        # Train the model
                        model.fit(X_train, y_train)

                        # Evaluate the model
                        accuracy = model.score(X_test, y_test)

                        # Store the model in session state
                        st.session_state.model = model
                        st.session_state.training_complete = True

                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")

                        # Save the model to file (optional for demo persistence)
                        joblib.dump(model, 'textnow_topic_model.pkl')
                        st.write("Model saved to disk for future use")

                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")

            # When no real data is available, use simulated data
            if st.button("Use Demo Data (for Hackathon)"):
                with st.spinner("Creating demo data..."):
                    # Create synthetic training data for demo purposes
                    demo_data = []

                    # Technical Support
                    support_messages = [
                        "My app keeps crashing when I make calls",
                        "Can't send text messages through the app",
                        "Having issues with my TextNow service",
                        "My dogfood device isn't receiving notifications",
                        "App freezes when switching between tabs",
                        "How do I fix the audio quality issues?",
                        "Can't log into my account after update",
                        "My messages aren't being delivered",
                        "Battery drains quickly when using the app",
                        "Getting error code 403 when trying to call"
                    ]

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
                        "Seeing unusual number of chargebacks",
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

                    # iOS
                    ios_messages = [
                        "SwiftUI implementation for the new chat screen",
                        "iOS app crashes when accessing media library",
                        "Need to update for iOS 16 compatibility",
                        "Memory leak in the iOS call screen",
                        "How do we implement PushKit for VoIP?",
                        "App Store rejected our latest build",
                        "Need to optimize battery usage in background",
                        "iPhone 14 Pro display issues with our app",
                        "Implementing CallKit integration for iOS",
                        "Swift package dependencies need updating"
                    ]

                    # Android
                    android_messages = [
                        "Implementing Material Design 3 components",
                        "Android app crashes on Samsung devices",
                        "Need to fix ANR in the messaging thread",
                        "How do we implement this new Android API?",
                        "Firebase messaging not working on Android 12",
                        "Memory optimization for low-end Android devices",
                        "Google Play policy compliance issue",
                        "Implementing background service for notifications",
                        "Android battery optimization breaking push",
                        "Need to update target SDK version"
                    ]

                    # Combine all messages with their topics
                    message_sets = [
                        (support_messages, "Technical Support and Troubleshooting", "#dogfood_help"),
                        (data_messages, "Data Analysis and Insights", "#ask-data-team"),
                        (fraud_messages, "Fraud Detection and Prevention", "#fraud-squad"),
                        (sre_messages, "Site Reliability and Infrastructure", "#sre"),
                        (backend_messages, "Backend Systems and Development", "#backend"),
                        (ios_messages, "iOS Development and Issues", "#ios"),
                        (android_messages, "Android Development and Issues", "#android")
                    ]

                    # Create records
                    for messages, topic, channel in message_sets:
                        for msg in messages:
                            timestamp = datetime.datetime.now() - datetime.timedelta(days=30) + datetime.timedelta(
                                days=30 * len(demo_data) / 70)
                            demo_data.append({
                                'message': msg,
                                'channel_id': channel,
                                'timestamp': timestamp,
                                'topic': topic
                            })

                    # Convert to DataFrame
                    demo_df = pd.DataFrame(demo_data)

                    # Store in session state
                    st.session_state.df = demo_df
                    st.session_state.data_loaded = True

                    # Display the data
                    st.write(f"Created {len(demo_df)} demo messages")
                    st.dataframe(demo_df[['message', 'channel_id', 'topic']].head(10))

                    # Show data distribution
                    st.subheader("Message Distribution by Channel")
                    channel_counts = demo_df['channel_id'].value_counts().reset_index()
                    channel_counts.columns = ['Channel', 'Count']
                    st.bar_chart(channel_counts.set_index('Channel')['Count'])

                    # Set row count for demo
                    st.session_state.row_count = 12345

                    # Auto-train the model with demo data
                    try:
                        # Split into train and test sets
                        X_train, X_test, y_train, y_test = train_test_split(
                            demo_df['message'], demo_df['topic'], test_size=0.2, random_state=42
                        )

                        # Create pipeline with TF-IDF and Naive Bayes
                        model = Pipeline([
                            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                            ('classifier', MultinomialNB())
                        ])

                        # Train the model
                        model.fit(X_train, y_train)

                        # Evaluate the model
                        accuracy = model.score(X_test, y_test)

                        # Store the model in session state
                        st.session_state.model = model
                        st.session_state.training_complete = True

                        st.success(f"Demo model trained successfully! Accuracy: {accuracy:.2%}")

                    except Exception as e:
                        st.error(f"Error training demo model: {str(e)}")

# Classifier Tab
with tab1:
    st.header("Slack-Channel Message Classification")

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
            st.info(
                "Please go to the 'Data Management' tab to load data and train the model first, or use the demo data option.")

        # Classification button
        classify_button = st.button("Classify Message")
        if classify_button and user_input and st.session_state.training_complete:
            with st.spinner("Analyzing message..."):
                # Get the model
                model = st.session_state.model

                # Perform classification
                topic = model.predict([user_input])[0]
                probas = model.predict_proba([user_input])[0]

                # Get the class indices to match with probabilities
                classes = model.classes_

                # Create results dictionary
                result = {
                    "topic": topic,
                    "channel": reverse_channel_mapping[topic],
                    "confidence": max(probas),
                    "all_topics": classes,
                    "all_confidences": probas
                }

                # Add to history
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                history_item = {
                    "time": timestamp,
                    "message": user_input[:50] + "..." if len(user_input) > 50 else user_input,
                    "topic": result["topic"],
                    "channel": result["channel"],
                    "confidence": result["confidence"],
                    "full_message": user_input,
                    "all_topics": result["all_topics"],
                    "all_confidences": result["all_confidences"]
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
                <h4 style="margin-top:0;">Recommended Channel: <span style="color:#ff4b4b;">{latest['topic']}</span></h4>
                <p><strong>Slack-Channel-id:</strong> {latest['channel']}<br/>
                <strong>Confidence:</strong> {latest['confidence']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

            # Show confidence for all channels
            st.subheader("All Channels")

            # Create a DataFrame for display
            results_data = []
            for topic, confidence in zip(latest['all_topics'], latest['all_confidences']):
                channel = reverse_channel_mapping[topic]
                results_data.append({
                    'Channel': topic,
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
    #     "Technical Issue": "I'm having trouble with my TextNow app. It crashes every time I try to send a picture message.",
    #     "Data Question": "Could someone help me understand the recent drop in active users? I need to analyze the trend.",
    #     "Fraud Alert": "We're seeing an unusual pattern of sign-ups from the same IP range. Possible account farming?",
    #     "Outage Report": "The messaging service seems to be down. Multiple users reporting delivery failures.",
    #     "Backend Question": "How do I modify the rate limiting for the messaging API? The current settings are too restrictive.",
    #     "iOS Bug": "The latest iOS build has a UI glitch when switching between dark and light mode.",
    #     "Android Feature": "Is there a way to implement the new notification system in the Android app without breaking compatibility?"
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