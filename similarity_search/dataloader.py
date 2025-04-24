"""
Data Loader Module for Similarity Search

This module provides functionality to load and preprocess conversation data for similarity search.
It handles data loading from CSV files and Slack conversations, with support for anonymization
and formatting of messages.

Key Features:
- CSV and Slack conversation data loading
- Message preprocessing and cleaning
- User and channel anonymization
- Conversation formatting
- Data validation and error handling

Example:
    To use the data loader:
    >>> loader = ChatDataLoader(file_path)
    >>> df = loader.preprocess_training_data()
    >>> conversations = await loader.load_conversations(slack_app, channel_id, thread_id)
"""

from similarity_search.random_names import get_random_names
import pandas as pd
import re 
import html
from slack_bolt.async_app import AsyncApp
import os
from typing import Dict

class ChatDataLoader:
    """
    A class for loading and preprocessing conversation data.
    
    This class handles the loading and preprocessing of conversation data from both
    CSV files and Slack conversations. It provides functionality for data cleaning,
    anonymization, and formatting.
    
    Attributes:
        data (pd.DataFrame): Raw conversation data
        user_map (Dict[str, str]): Mapping of user IDs to anonymized names
        channel_map (Dict[str, str]): Mapping of channel IDs to anonymized names
        user_group_map (Dict[str, str]): Mapping of user group IDs to anonymized names
        random_names (List[str]): List of random names for anonymization
    """
    
    def __init__(self) -> None:
        """
        Initialize the data loader.
        
        """
        self.user_map: Dict[str, str] = {}
        self.channel_map: Dict[str, str] = {}
        self.user_group_map: Dict[str, str] = {}
        self.random_names = random_names

    def preprocess_training_data(self, file_path: str) -> pd.DataFrame:
        """
        Preprocess the training data from the CSV file.

        Args:
            file_path (str): Path to the raw data file. Must contain columns:
                - User: User identifier
                - Message: Message content
                - Channel ID: Channel identifier
                - Thread ID: Thread identifier
                - Timestamp: Message timestamp
        
        Returns:
            pd.DataFrame: Preprocessed conversation data
        """
        return self.preprocess_data(pd.read_csv(file_path))
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data by cleaning and formatting messages.
        
        Args:
            data (pd.DataFrame): Input data to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed data with cleaned messages and formatted conversations
        """
        # Load random names for anonymization
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        random_names_path = os.path.join(project_root, 'similarity_search', 'data', 'random_names.txt')
        self.random_names = random_names

        # Clean data - create a copy to avoid chained indexing warnings
        data = data.dropna().copy()
        
        # Convert timestamp - handle both Unix epoch and regular datetime formats
        try:
            # Try regular datetime parsing first
            data.loc[:, "Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True)
        except:
            # If that fails, assume Unix epoch format
            data.loc[:, "Timestamp"] = pd.to_datetime(data["Timestamp"].astype(float), unit='s', utc=True)
        
        # Clean messages
        data.loc[:, 'cleaned_message'] = data['Message'].apply(self.clean_message)

        # Convert rows to conversations
        data = data.groupby(by=['Channel ID', 'Thread ID'], group_keys=False).apply(self.format_conversation).reset_index(name='conversation')
        return data

    def format_conversation(self, group: pd.DataFrame) -> str:
        """
        Format a group of messages into a conversation string.
        
        Args:
            group (pd.DataFrame): Group of messages to format
            
        Returns:
            str: Formatted conversation string
        """
        # Sort messages by timestamp
        messages = []
        group = group.sort_values(by='Timestamp', ascending=True)
        
        # Format each message
        for _, row in group.iterrows():
            user_id = row['User']
            if user_id not in self.user_map:
                self.user_map[user_id] = self.random_names.pop()
            user = self.user_map[user_id]
            messages.append(f"{user}: {row['cleaned_message']}")
        return '\n'.join(messages)
    
    def get_channel_name(self, channel_id: str) -> str:
        """
        Get or create an anonymized name for a channel.
        
        Args:
            channel_id (str): Channel ID to look up
            
        Returns:
            str: Anonymized channel name
        """
        if channel_id in self.channel_map:
            return self.channel_map[channel_id]
        else:
            name = f"[Channel {len(self.channel_map ) + 1}]"
            self.channel_map[channel_id] = name
            return name

    def clean_message(self, message: str) -> str:
        """
        Clean and anonymize a message.
        
        This method handles various Slack message formatting and anonymization:
        - Channel mentions
        - User mentions
        - User groups
        - URLs
        - Emails
        - Code blocks
        - System messages
        - Text formatting
        
        Args:
            message (str): Message to clean
            
        Returns:
            str: Cleaned and anonymized message
        """
        def replace_channel(match): 
            channel_id = match.group(1)
            if channel_id in self.channel_map:
                return self.channel_map[channel_id]
            else:
                name = f"[Channel {len(self.channel_map ) + 1}]"
                self.channel_map[channel_id] = name
                return name

        def replace_mention(match):
            user_id = match.group(1)
            if user_id in self.user_map:
                return f"@{self.user_map[user_id]}"
            else:
                name = self.random_names.pop()
                self.user_map[user_id] = name
                return f"@{name}"
            
        def replace_user_group(match):
            user_group = match.group(1)
            if user_group in self.user_group_map:
                return f"{self.user_group_map[user_group]}"
            else:
                name = f"@[User Group {len(self.user_group_map) + 1}]"
                self.user_group_map[user_group] = name
            
        # Unescape HTML
        text = html.unescape(message)

        # Trim whitespace
        text = text.strip()

        # Replace various Slack message elements
        text = re.sub(r"<@([A-Z0-9]+)>", replace_mention, text)
        text = re.sub(r"<(https?://[^>]+)>", "[URL]", text)
        text = re.sub(r"\S+@\S+", "[EMAIL]", text)
        text = re.sub(r"<#(C[0-9A-Z]+)(?:\|[^>]*)?>", replace_channel, text)
        text = re.sub(r"```(.*?)```", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"`(.*?)`", r"\1", text)
        text = re.sub(r"<![a-z]+>", "[system:channel-notification]", text)
        text = re.sub(r"<!subteam\^([A-Z0-9]+)>", replace_user_group, text)

        # Handle text formatting
        text = re.sub(r"\*(.*?)\*", r"\1", text)  # *bold*
        text = re.sub(r"_(.*?)_", r"\1", text)    # _italic_
        text = re.sub(r"~(.*?)~", r"\1", text)    # ~strikethrough~

        return text
    
    async def load_conversations(self, slack_app: AsyncApp, channel_id: str, thread_id: str) -> pd.DataFrame:
        """
        Load conversations from a single channel thread in Slack.
        
        Args:
            slack_app (AsyncApp): Slack app instance
            channel_id (str): Channel ID to load from
            thread_id (str): Thread ID to load from
            
        Returns:
            pd.DataFrame: DataFrame containing the loaded conversations
        """
        # Get conversation from Slack
        conversation = await slack_app.client.conversations_replies(channel=channel_id, ts=thread_id)
        
        # Convert messages to DataFrame
        messages = []
        for msg in conversation["messages"]:
            slack_message = {
                "User": msg.get("user", ""),
                "Message": msg.get("text", ""),
                "Channel ID": channel_id,
                "Thread ID": thread_id,
                "Timestamp": pd.to_datetime(float(msg.get("ts", 0)), unit='s', utc=True)
            }
            messages.append(slack_message)
        
        return pd.DataFrame(messages)

random_names = ['Mckenna', 'Peter', 'Savanna', 'Bailey', 'Alexis', 'Riley', 'Hope', 'Alana', 'Jesse', 'Alec', 'Briana', 'Martin', 'Justin', 'Caitlyn', 'Francisco', 'Mikayla', 'Elise', 'Bryce', 'John', 'Devin', 'Ian', 'Daniel', 'Henry', 'Jada', 'Logan', 'Alyssa', 'Meagan', 'Amelia', 'Kyle', 'Olivia', 'Christina', 'Brian', 'Dustin', 'Johnathan', 'Abigail', 'Marcos', 'Tiffany', 'Victoria', 'Julio', 'Benjamin', 'Lorenzo', 'Levi', 'Edward', 'Richard', 'Connor', 'Russell', 'Kira', 'Dominic', 'Timothy', 'Carter', 'Shannon', 'Maggie', 'Joanna', 'Jade', 'Kristopher', 'Brandon', 'Hayley', 'Isaac', 'Sydney', 'Rosa', 'Haley', 'Kelly', 'Serena', 'Audrey', 'Jordan', 'Brayden', 'Gabriella', 'Emily', 'Isabelle', 'Kylie', 'Colin', 'Jeffery', 'Mary', 'Andrew', 'Mitchell', 'Dallas', 'Monica', 'Christine', 'Philip', 'Ashton', 'Grant', 'Trenton', 'Marc', 'Brenda', 'Clayton', 'Katlyn', 'Phillip', 'Drake', 'Alejandro', 'Karen', 'Miguel', 'Tommy', 'Robert', 'Caitlin', 'Chloe', 'Mackenzie', 'Nikolas', 'Christopher', 'Morgan', 'Samuel', 'Ronald', 'Logan', 'Lindsey', 'Sadie', 'Abby', 'Dante', 'Kaitlyn', 'Arturo', 'Krystal', 'Aaliyah', 'Natasha', 'Gary', 'Manuel', 'Douglas', 'Raul', 'Kristina', 'Jillian', 'Angelica', 'Jeremiah', 'Peyton', 'Alexus', 'Hunter', 'Vincent', 'Donald', 'Savannah', 'Sidney', 'Kylee', 'Alexia', 'Keith', 'Gina', 'Edgar', 'Jalen', 'Jarrett', 'Selena', 'Colton', 'Patricia', 'Tiana', 'Quentin', 'Mallory', 'Kyla', 'Cynthia', 'Nancy', 'Cesar', 'Curtis', 'Nolan', 'Jose', 'Armando', 'Marcus', 'Samantha', 'Lillian', 'Naomi', 'Rebecca', 'Ashley', 'Mya', 'Jennifer', 'Isabella', 'Julia', 'Diego', 'Jared', 'Dakota', 'Julianna', 'Lucas', 'Male name', 'Skylar', 'Adrianna', 'Conner', 'Noah', 'Jacob', 'Sebastian', 'Andrea', 'Braden', 'Margaret', 'Miranda', 'Derek', 'Jerry', 'Madeline', 'Brent', 'Riley', 'Mariah', 'Emmanuel', 'Mercedes', 'Ramon', 'Kara', 'Laura', 'Sergio', 'Skyler', 'Collin', 'Shelby', 'Kaitlin', 'Alicia', 'Dakota', 'Theodore', 'Mathew', 'Nickolas', 'Leah', 'George', 'Delaney', 'Ivan', 'Parker', 'Jenna', 'Katie', 'Ariana', 'Angel', 'Jasmin', 'Nathaniel', 'Desiree', 'Faith', 'Brittney', 'Matthew', 'Devon', 'Troy', 'Frank', 'Lisa', 'Thomas', 'Jake', 'Alexa', 'Kendra', 'Alex', 'Jocelyn', 'Catherine', 'Jackson', 'Josue', 'Chase', 'Raven', 'Brenna', 'Female name', 'Esmeralda', 'Bailey', 'Alexis', 'Allison', 'Garrett', 'Diamond', 'Karla', 'Hector', 'Avery', 'Melissa', 'Jarod', 'Sofia', 'Erik', 'Cassidy', 'Sierra', 'Jaime', 'Bryan', 'Erica', 'Brianna', 'Adam', 'Katelyn', 'Scott', 'Carson', 'Marco', 'Melanie', 'Kendall', 'Heather', 'Antonio', 'Megan', 'Ciara', 'Guadalupe', 'Andy', 'Danielle', 'Eduardo', 'Leslie', 'Tessa', 'Stephanie', 'Jorge', 'Dana', 'Anne', 'Celeste', 'Miles', 'Xavier', 'Elias', 'Dominick', 'Roberto', 'Nicole', 'Bianca', 'James', 'Ricardo', 'Lawrence', 'Cheyenne', 'Callie', 'Brandi', 'William', 'Molly', 'Griffin', 'Skyler', 'Aubrey', 'Destiny', 'Cooper', 'Lauren', 'Jimmy', 'Chandler', 'Grace', 'Carl', 'Marisa', 'Cindy', 'Julian', 'Jordan', 'Damon', 'Camille', 'Rose', 'Leonardo', 'Alissa', 'Lydia', 'Ryan', 'Tanner', 'Juliana', 'Amy', 'Dalton', 'Giovanni', 'Kaylee', 'Cameron', 'Arthur', 'Katrina', 'Trevor', 'Katherine', 'Charles', 'Evelyn', 'Brenden', 'Michael', 'Zoe', 'Natalia', 'Katelynn', 'Jakob', 'Darren', 'Gavin', 'Madison', 'Wyatt', 'Dylan', 'Allen', 'Landon', 'Erick', 'Johnny', 'Rachael', 'Bethany', 'Kate', 'Kameron', 'Trey', 'Spencer', 'Mario', 'Sophie', 'Makenzie', 'Karina', 'Nicolas', 'Aidan', 'Daniela', 'Sabrina', 'Colby', 'Allyson', 'Caroline', 'Paul', 'Joshua', 'Travis', 'Ty', 'Emma', 'Cody', 'Breanna', 'Claudia', 'Zachery', 'Ruben', 'Sarah', 'Carlos', 'Priscilla', 'Elijah', 'Mckenzie', 'Cole', 'Jesus', 'Josiah', 'Randy', 'Brett', 'Javier', 'Jonathan', 'Jonathon', 'Kobe', 'Dillon', 'Ethan', 'Jeremy', 'Kirsten', 'Tabitha', 'Casey', 'Tara', 'Morgan', 'Hannah', 'Wesley', 'Maya', 'Jeffrey', 'Eli', 'Payton', 'Joseph', 'Brock', 'Darius', 'Alexander', 'Yesenia', 'Deja', 'Brendan', 'Anastasia', 'Adrian', 'Bradley', 'Oscar', 'Chad', 'Fernando', 'Tristen', 'Gregory', 'Ricky', 'Michaela', 'Meghan', 'Kennedy', 'Angela', 'Jasmine', 'Owen', 'Whitney', 'Jack', 'Shane', 'Shania', 'Alondra', 'Dawson', 'Erin', 'Isabel', 'Jamie', 'Shawn', 'Derrick', 'Evan', 'Jazmine', 'Angel', 'Israel', 'Cameron', 'Kiana', 'Crystal', 'Caleb', 'Madeleine', 'Austin', 'Omar', 'Brennan', 'Kevin', 'Ana', 'Andres', 'Brandy', 'Alexandria', 'Gage', 'Deanna', 'Alejandra', 'Joe', 'Amanda', 'Dennis', 'Erika', 'Isaiah', 'Larry', 'Rafael', 'Ashlee', 'Cory', 'Joel', 'Casey', 'Kaleb', 'Ashleigh', 'Jason', 'Cecilia', 'Andre', 'Alexandra', 'Gerardo', 'Lily', 'Kyra', 'Arianna', 'Kiara', 'Jacqueline', 'Taylor', 'Kayla', 'Gabriel', 'Jazmin', 'Tristan', 'Nicholas', 'Makayla', 'Hailey', 'Patrick', 'Ashlyn', 'Sandra', 'Deandre', 'Edwin', 'Lane', 'Victor', 'Marissa', 'Courtney', 'Brooke', 'Kristin', 'Tatiana', 'Albert', 'Corey', 'Monique', 'Nina', 'Harrison', 'Luke', 'Zackary', 'Seth', 'Hayden', 'Nathan', 'Lindsay', 'Aaron', 'Carolina', 'Daisy', 'Christian', 'Elizabeth', 'Natalie', 'Autumn', 'Diana', 'Summer', 'Gianna', 'Micheal', 'Valerie', 'Gabriela', 'Damian', 'Genesis', 'Claire', 'Micah', 'Maria', 'Chance', 'Bridget', 'Kenneth', 'Zane', 'Donovan', 'April', 'Cassandra', 'Carmen', 'Kassandra', 'Louis', 'Valeria', 'Kathleen', 'Stephen', 'Luis', 'Maxwell', 'Trent', 'Juan', 'Kathryn', 'Avery', 'Blake', 'Calvin', 'Paige', 'Alison', 'Jessica', 'Anna', 'Ruby', 'Amber', 'Madelyn', 'Mikaela', 'Taylor', 'Julie', 'Pedro', 'Mark', 'Alfredo', 'Cristian', 'Brittany', 'Michelle', 'Peyton', 'Kelsey', 'Enrique', 'Liam', 'Anthony', 'David', 'Preston', 'Adriana', 'Mia', 'Drew', 'Hanna', 'Rebekah', 'Vanessa', 'Angelina', 'Steven', 'Chelsea', 'Alberto', 'Max', 'Imani', 'Malik', 'Kimberly', 'Jonah', 'Tori', 'Mason', 'Kailey', 'Jordyn', 'Bryanna', 'Brooklyn', 'Eric', 'Meredith', 'Asia', 'Danny', 'Kristen', 'Elena', 'Raymond', 'Giselle', 'Rachel', 'Holly', 'Veronica', 'Abraham', 'Dominique', 'Tyler', 'Carly', 'Gabrielle', 'Sean', 'Alan', 'Damien', 'Brady', 'Tony', 'Sophia', 'Sara', 'Ariel', 'Zachary', 'Cierra']
